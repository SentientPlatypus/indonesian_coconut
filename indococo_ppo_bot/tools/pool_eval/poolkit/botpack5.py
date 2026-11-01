"""Adapters for RLBot v5 botpack bots (github.com/RLBot/botpack, Nov 2026).

- Ripple (Rolv-Arild/Ripple-RLBot): runs the bot's own `obs.py`, `act.py` and
  `agent.py` (ONNX policy + BC base model) on an RLGym v1 view of the state,
  tick skip 4, with its 5-decision delayed action queue. The RLBot bot picks
  an action-selection mode per episode with a bandit; here the mode is fixed
  (default `kl`, the bandit's first arm) so results do not depend on history.
- GigaLearn "GGLBot" bots (Kira, Dice, Chronos): C++ bots, so the obs builder
  (`AdvancedObs` / `AdvancedObsPadded`) and `DefaultAction` (90 actions + mask)
  are ported line by line from the bot's `inc/RLGymCPP`. The TorchScript
  `SHARED_HEAD.lt` and `POLICY.lt` load as shipped. Tick skip 8 with action
  delay 7: the previous action is held for 7 ticks, the new one for the 8th,
  as in GigaLearn training and the bot's `RLBotClient.cpp`. Greedy (argmax
  over the masked logits), as `InferAction(..., deterministic=true)`.

Sim state (has_jumped, air_time_since_jump, pad timers) comes from RocketSim
directly, which is what these bots saw in training; their RLBot clients
approximate some of it from packets.
"""
from __future__ import annotations

import os

import numpy as np

from .agent import Agent
from .bots import load_module
from .v1compat import install_stub_modules, players_for, to_v1
from .view import AgentView

DOUBLEJUMP_MAX_DELAY = 1.25
INV = np.array([-1.0, -1.0, 1.0])


class Ripple(Agent):
    tick_skip = 4

    def __init__(self, directory: str, variant: str = "v1.1", mode: str = "kl") -> None:
        super().__init__()
        install_stub_modules()
        src = os.path.join(directory, "src")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        self._obs_mod = load_module(os.path.join(src, "obs.py"), "ripple_obs")
        self._act_mod = load_module(os.path.join(src, "act.py"), "ripple_act")
        agent_mod = load_module(os.path.join(src, "agent.py"), "ripple_agent")
        bots = os.path.join(directory, "bots", variant)
        policy = os.path.join(bots, f"ripple-{variant}.onnx")
        base = os.path.join(bots, f"ripple-{variant}-bcm.onnx")
        self.agent = agent_mod.Agent(policy, base, mode, backend="onnx", seed=0)
        self.mode = mode

    def reset(self, view: AgentView) -> None:
        super().reset(view)
        self.agent.rng = np.random.default_rng(np.random.randint(2**31))
        self.queue = np.zeros((6, 1, 8))
        self.obs = self._obs_mod.RippleObs(self.queue)
        self.parser = self._act_mod.RippleAction(self.queue)
        gs, _ = self._v1(view)
        self.obs.reset(gs)
        self.actions = np.nan_to_num(self.queue[-1], nan=0.0).copy()

    def _v1(self, view: AgentView):
        gs, me = to_v1(view, self.tick_skip)
        gs.players = players_for(gs, me, sort_by_ball=False)
        return gs, me

    def decide(self, view: AgentView) -> np.ndarray:
        gs, _ = self._v1(view)
        self.obs.pre_step(gs, 1, elapsed_ticks=self.tick_skip)
        self.obs.prev_prev_actions[:1] = self.actions
        stochastic = bool(np.all(gs.ball.position[:2] == 0))  # stochastic kickoffs
        chosen = self.agent.act(self.obs.current_obs, stochastic=stochastic)
        self.actions = self.parser.parse_actions(chosen, gs)
        return np.asarray(self.actions[0], dtype=np.float64)


def gigalearn_action_table():
    """DefaultAction.cpp: 90 rows plus the ground/air/jump/boost masks."""
    rows, ground = [], 0
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    rows.append([throttle, steer, 0, steer, 0, 0, boost, handbrake])
    ground = len(rows)
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        hb = float(jump == 1 and (pitch != 0 or yaw != 0 or roll != 0))
                        rows.append([boost, yaw, pitch, yaw, roll, jump, boost, hb])
    a = np.array(rows, dtype=np.float64)
    n = len(a)
    idx = np.arange(n)
    jump_m = a[:, 5] == 1
    boost_m = a[:, 6] == 1
    ground_m = idx < ground
    air_m = (idx > ground) & ~jump_m  # sic: `i > numGroundActions` in the C++
    air_m |= ground_m & (a[:, 0] == a[:, 6]) & ((a[:, 3] != 0) == (a[:, 7] == 1))
    return a, ground_m, air_m, jump_m, boost_m


def load_gigalearn_seq(path: str, activation: str):
    """GigaLearn saves its Sequential as a TorchScript module without `forward`.

    Rebuilds it from the parameters: Linear -> LayerNorm -> activation per
    hidden layer (Models.h), activation set in the bot's RLBotMain.cpp.
    """
    import torch

    act = {"relu": torch.nn.ReLU, "leaky_relu": torch.nn.LeakyReLU}[activation]
    ts = torch.jit.load(path, map_location="cpu")
    seq = torch.nn.Sequential()
    for _, mod in list(ts.named_modules())[1:]:
        params = dict(mod.named_parameters(recurse=False))
        if not params:
            seq.append(act())
            continue
        w, b = params["weight"], params["bias"]
        layer = torch.nn.Linear(w.shape[1], w.shape[0]) if w.dim() == 2 else torch.nn.LayerNorm(w.shape[0])
        layer.load_state_dict({"weight": w.detach().clone(), "bias": b.detach().clone()})
        seq.append(layer)
    return seq.eval()


def has_flip_or_jump(car) -> bool:
    return bool(car.on_ground or (not car.has_flipped and not car.has_double_jumped
                                  and car.air_time_since_jump < DOUBLEJUMP_MAX_DELAY))


class GigaLearn(Agent):
    """GGLBot: AdvancedObs[Padded] + DefaultAction, shared head + policy TorchScript."""

    tick_skip = 8
    action_delay = 7

    def __init__(self, directory: str, padded: int = 0, activation: str = "relu",
                 model_dir: str = "rlbot") -> None:
        super().__init__()
        d = os.path.join(directory, model_dir)
        self.shared = load_gigalearn_seq(os.path.join(d, "SHARED_HEAD.lt"), activation)
        self.policy = load_gigalearn_seq(os.path.join(d, "POLICY.lt"), activation)
        self.padded = padded
        self.table, self.m_ground, self.m_air, self.m_jump, self.m_boost = gigalearn_action_table()

    def reset(self, view: AgentView) -> None:
        super().reset(view)
        self.prev_action = np.zeros(8)  # controls currently applied (C++ st.controls)
        self.new_action = np.zeros(8)
        self.phase = 0  # 0: decide at next call; 1: apply new action at next call

    @staticmethod
    def _phys(p, inv: bool):
        pos, vel, ang = (np.asarray(x, dtype=np.float64) for x in (p.position, p.linear_velocity, p.angular_velocity))
        rot = np.asarray(p.rotation_mtx, dtype=np.float64)  # columns forward, right, up
        if inv:
            pos, vel, ang, rot = pos * INV, vel * INV, ang * INV, rot * INV[:, None]
        return pos, vel, ang, rot

    def _player_obs(self, car, inv, bpos, bvel) -> list:
        pos, vel, ang, rot = self._phys(car.physics, inv)
        return [*(pos / 5000), *rot[:, 0], *rot[:, 2], *(vel / 2300), *(ang / 3),
                *(rot.T @ ang / 3), *(rot.T @ (bpos - pos) / 5000), *(rot.T @ (bvel - vel) / 2300),
                car.boost_amount / 100, float(car.on_ground), float(has_flip_or_jump(car)),
                float(car.is_demoed), float(car.has_jumped)]

    def build_obs(self, view: AgentView) -> np.ndarray:
        s, me = view.state, view.car
        inv = me.team_num == 1
        bpos, bvel, bang, _ = self._phys(s.ball, inv)
        timers = np.asarray(s.inverted_boost_pad_timers if inv else s.boost_pad_timers, dtype=np.float64)
        pads = np.where(timers == 0, 1.0, 1.0 / (1.0 + np.maximum(timers, 0)))
        if self.padded:
            pads = np.where(timers == 0, 1.0, np.minimum(pads, 0.999))
        obs = [*(bpos / 5000), *(bvel / 2300), *(bang / 3), *self.prev_action, *pads]
        obs += self._player_obs(me, inv, bpos, bvel)
        mates, opps = [], []
        for cid, car in s.cars.items():
            if cid == view.car_id:
                continue
            (mates if car.team_num == me.team_num else opps).append(self._player_obs(car, inv, bpos, bvel))
        if self.padded:
            size = 29
            mates += [[0.0] * size] * (self.padded - 1 - len(mates))
            opps += [[0.0] * size] * (self.padded - len(opps))
            np.random.shuffle(mates)
            np.random.shuffle(opps)
        for p in mates + opps:
            obs += p
        return np.asarray(obs, dtype=np.float32)

    def mask(self, car) -> np.ndarray:
        m = self.m_ground.copy() if car.on_ground else self.m_air.copy()
        if car.boost_amount == 0:
            m &= ~self.m_boost
        if has_flip_or_jump(car):
            m |= self.m_jump
        return m

    def decide(self, view: AgentView) -> np.ndarray:
        import torch

        obs = torch.from_numpy(self.build_obs(view))[None]
        with torch.inference_mode():
            logits = self.policy(self.shared(obs))[0].numpy()
        m = self.mask(view.car)
        if not m.any():
            m[:] = True
        logits = np.where(m, logits, -np.inf)
        return self.table[int(np.argmax(logits))]

    def controls(self, view: AgentView, tick: int) -> np.ndarray:
        if tick >= self._next_decision:
            if self.phase == 0:
                self.new_action = self.decide(view)
                self.phase = 1
                self._next_decision = tick + self.action_delay
                self._held = self.prev_action
            else:
                self.prev_action = self.new_action
                self.phase = 0
                self._next_decision = tick + (self.tick_skip - self.action_delay)
                self._held = self.prev_action
        return self._held


BIG_PADS = (3, 4, 15, 18, 29, 30)
PRED_TICKS = (30, 60, 120, 180, 240)


class Galactus(GigaLearn):
    """GalactusV2: GalactusOBS(3) = AdvancedObsPadded(3) without the 0.999 pad clamp,
    plus the nearest active big pad within 1500 uu and 5 ball-prediction samples
    (+0.25/0.5/1/1.5/2 s). Predictions come from a ball-only RocketSim arena, as
    RocketSim's BallPredTracker did in training."""

    def __init__(self, directory: str, activation: str = "leaky_relu", model_dir: str = "rlbot") -> None:
        super().__init__(directory, padded=3, activation=activation, model_dir=model_dir)
        import RocketSim as rs
        from rlgym.rocket_league.common_values import BOOST_LOCATIONS

        self._rs = rs
        self._pads_xyz = np.asarray(BOOST_LOCATIONS, dtype=np.float64)
        self._arena = None  # created on first use, after the match engine has run RocketSim.init

    def _ball_pred(self, ball) -> list[np.ndarray]:
        rs = self._rs
        if self._arena is None:
            self._arena = rs.Arena(rs.GameMode.SOCCAR)
        self._arena.ball.set_state(rs.BallState(pos=rs.Vec(*map(float, ball.position)),
                                                vel=rs.Vec(*map(float, ball.linear_velocity)),
                                                ang_vel=rs.Vec(*map(float, ball.angular_velocity))))
        out, t = [], 0
        for target in PRED_TICKS:
            self._arena.step(target - t)
            t = target
            st = self._arena.ball.get_state()
            out.append(np.array([st.pos.x, st.pos.y, st.pos.z, st.vel.x, st.vel.y, st.vel.z]))
        return out

    def build_obs(self, view: AgentView) -> np.ndarray:
        s, me = view.state, view.car
        inv = me.team_num == 1
        bpos, bvel, bang, _ = self._phys(s.ball, inv)
        timers = np.asarray(s.inverted_boost_pad_timers if inv else s.boost_pad_timers, dtype=np.float64)
        avail = timers == 0
        pads = np.where(avail, 1.0, 1.0 / (1.0 + timers))
        obs = [*(bpos / 5000), *(bvel / 2300), *(bang / 3), *self.prev_action, *pads]
        mpos = self._phys(me.physics, inv)[0]
        nearest, best = np.zeros(3), 1500.0 ** 2
        for i in BIG_PADS:
            if not avail[i]:
                continue
            d2 = float(np.sum((self._pads_xyz[i, :2] - mpos[:2]) ** 2))
            if d2 <= best:
                best, nearest = d2, self._pads_xyz[i] - mpos
        obs += list(nearest / 5000)
        obs += self._player_obs(me, inv, bpos, bvel)
        mates, opps = [], []
        for cid, car in s.cars.items():
            if cid != view.car_id:
                (mates if car.team_num == me.team_num else opps).append(self._player_obs(car, inv, bpos, bvel))
        mates += [[0.0] * 29] * (2 - len(mates))
        opps += [[0.0] * 29] * (3 - len(opps))
        np.random.shuffle(mates)
        np.random.shuffle(opps)
        for p in mates + opps:
            obs += p
        for pv in self._ball_pred(s.ball):
            if inv:
                pv = pv * np.concatenate([INV, INV])
            obs += list(pv[:3] / 5000) + list(pv[3:] / 2300)
        return np.asarray(obs, dtype=np.float32)
