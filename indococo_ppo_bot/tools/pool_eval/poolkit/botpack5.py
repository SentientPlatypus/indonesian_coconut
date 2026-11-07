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
        """Real ByteAction.cpp:114-118 restores jump when HasFlipOrJump OR contact normal.z > 0.9.
        Fallback from RLBotClient.cpp:73-78: upside-down on floor with low velocity."""
        m = self.m_ground.copy() if car.on_ground else self.m_air.copy()
        if car.boost_amount == 0:
            m &= ~self.m_boost
        # Turtle recovery: restore jump if has flip or upside-down on floor
        has_jump = has_flip_or_jump(car)
        contact_up = (
            not car.is_demoed
            and not car.on_ground
            and car.physics.rotation_mtx[2, 2] < -0.9
            and 0 <= car.physics.position[2] < 60
            and abs(car.physics.linear_velocity[2]) < 50
        )
        if has_jump or contact_up:
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


class Megabyte(GigaLearn):
    """Megabyte (SubparN0va/ByteBotpack, MIT 2026): ByteObs (233) + DefaultAction.
    
    233 obs: ball (9), prev action (8), 34 pads, teammate count / 2, opponent count / 3,
    self (30), 2 teammate slots (30 each, zero padded), 3 opponent slots (30 each, zero padded).
    Player (30): pos, forward, up, vel, angvel, local angvel, local ball pos/vel,
    boost, on_ground, has_flip, is_demoed, has_jumped, padding 1.0.
    Orange mirrors x and y (NOT z). Pad timers use the OPPOSITE orientation from
    the player-relative pads (legacy behavior).
    """
    def __init__(self, directory: str) -> None:
        super().__init__(directory, padded=0, activation="relu", model_dir="")

    def build_obs(self, view: AgentView) -> np.ndarray:
        s, me = view.state, view.car
        inv = me.team_num == 1
        # _phys already applies INV=[-1,-1,1] for orange, matching ByteObs C++ InvertPhys
        bpos, bvel, bang, _ = self._phys(s.ball, inv)
        # Real ByteObs.cpp:70-94 uses player-relative pad ACTIVITY but opposite-relative TIMERS (legacy).
        # Use player-relative activity at index k, and opposite-relative timers at index k.
        # No additional index reversal - the arrays are already oriented.
        active_timers = s.inverted_boost_pad_timers if inv else s.boost_pad_timers
        opposite_timers = s.boost_pad_timers if inv else s.inverted_boost_pad_timers
        active_arr = np.asarray(active_timers, dtype=np.float64)
        timer_arr = np.asarray(opposite_timers, dtype=np.float64)
        pads_obs = np.where(active_arr == 0, 1.0, 1.0 / (1.0 + timer_arr))
        obs = [*(bpos / 5000), *(bvel / 2300), *(bang / 3), *self.prev_action, *pads_obs]
        
        # Real ByteObs.cpp:15-32,114-117 sorts by world-space distance squared with carId tie-break
        # Carry (carId, car) pairs, filter by team, sort by distance then carId
        mate_data = []
        opp_data = []
        for cid, car in s.cars.items():
            if cid != view.car_id:
                dist_sq = np.linalg.norm(np.array(car.physics.position) - np.array(s.ball.position))**2
                if car.team_num == me.team_num:
                    mate_data.append((cid, car, dist_sq))
                else:
                    opp_data.append((cid, car, dist_sq))
        
        # Sort by distance squared, then by carId, and truncate
        mate_data.sort(key=lambda x: (x[2], x[0]))
        opp_data.sort(key=lambda x: (x[2], x[0]))
        mates = [car for _, car, _ in mate_data[:2]]
        opps = [car for _, car, _ in opp_data[:3]]
        
        obs += [len(mates) / 2.0, len(opps) / 3.0]
        obs += self._player_obs(me, inv, bpos, bvel) + [1.0]  # self + padding
        mate_obs = [self._player_obs(c, inv, bpos, bvel) + [1.0] for c in mates]
        opp_obs = [self._player_obs(c, inv, bpos, bvel) + [1.0] for c in opps]
        mate_obs += [[0.0] * 30] * (2 - len(mate_obs))
        opp_obs += [[0.0] * 30] * (3 - len(opp_obs))
        np.random.shuffle(mate_obs)
        np.random.shuffle(opp_obs)
        for p in mate_obs + opp_obs:
            obs += p
        return np.asarray(obs, dtype=np.float32)


class Seer(Agent):
    """Seer (Neville Walo, RLBotPack v4 `RLBotPack/Seer`): 1v1 LSTM (512) policy.

    Runs the bot's own `helper.py` (Seer_Network, invert_*, impute_features,
    one-hot action encoding; needs numba + scikit-learn) on a v4-packet view of
    the RocketSim state, mirroring its `bot.py`:
    - packet fields: location, euler (pitch, yaw, roll; RLBot convention via
      RLGym's quat_to_euler), velocity, angular velocity, demo timer (seconds
      demolished, counting up), boost 0-100, has_wheel_contact, has_flip;
      orange plays inverted (bot index != 0).
    - pad `timer` as in RLBot v4: seconds since pickup, 0 when active
      (RocketSim keeps the countdown; converted with 4 s / 10 s respawn).
    - has_flip: not double_jumped (v4 sets it after a dodge too) and not
      (jumped and > 1.5 s since last wheel contact), wheel contact sampled at
      decisions like the bot does.
    - LSTM state and prev action reset per game; greedy actions; tick skip 8.
    """

    tick_skip = 8

    def __init__(self, directory: str) -> None:
        super().__init__()
        import torch

        self._helper = load_module(os.path.join(directory, "helper.py"), "seer_helper")
        self.model = self._helper.Seer_Network()
        self.model.load_state_dict(torch.load(os.path.join(directory, "Seer.pt"), map_location="cpu"))
        self.model.eval()

    def reset(self, view: AgentView) -> None:
        import torch

        super().reset(view)
        self.state = (torch.zeros(1, 1, 512), torch.zeros(1, 1, 512))
        self.prev_action = np.array([1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.last_contact = {}
        self.demo = {}
        self.last_tick = view.tick

    def _player(self, cid, car, tick, inverted) -> np.ndarray:
        p = car.physics
        pitch, yaw, roll = p.euler_angles
        t = tick / 120.0
        if car.on_ground or cid not in self.last_contact:
            self.last_contact[cid] = t
        timeout = (t - self.last_contact[cid]) > self._helper.flip_timeout and car.has_jumped
        double_jumped = car.has_double_jumped or car.has_flipped
        has_flip = (not double_jumped) and not timeout
        a = np.array([*p.position, pitch, yaw, roll, *p.linear_velocity, *p.angular_velocity,
                      self.demo.get(cid, 0.0), car.boost_amount, float(car.on_ground), float(has_flip)],
                     dtype=np.float32)
        return self._helper.invert_player_data(a) if inverted else a

    def decide(self, view: AgentView) -> np.ndarray:
        import torch

        s, h = view.state, self._helper
        dt = (view.tick - self.last_tick) / 120.0
        self.last_tick = view.tick
        for cid, car in s.cars.items():
            self.demo[cid] = self.demo.get(cid, 0.0) + dt if car.is_demoed else 0.0
        inverted = view.car.team_num == 1
        opp = next(c for c in s.cars if c != view.car_id)
        me_a = self._player(view.car_id, view.car, view.tick, inverted)
        op_a = self._player(opp, s.cars[opp], view.tick, inverted)
        cd = np.asarray(s.boost_pad_timers, dtype=np.float64)
        from rlgym.rocket_league.common_values import BOOST_LOCATIONS

        respawn = np.where(np.asarray(BOOST_LOCATIONS)[:, 2] > 72, 10.0, 4.0)
        pads = np.where(cd > 0, respawn - cd, 0.0).astype(np.float32)
        if inverted:
            pads = h.invert_boost_data(pads)
        b = s.ball
        ball = np.array([*b.position, *b.linear_velocity, *b.angular_velocity], dtype=np.float32)
        if inverted:
            ball = h.invert_ball_data(ball)
        enc = h.get_action_encoding(self.prev_action.reshape(1, -1))
        x = h.impute_features(me_a, op_a, pads, ball, enc.reshape(-1)).reshape(1, -1)
        with torch.no_grad():
            actions, _, _, self.state = self.model(torch.tensor(x, dtype=torch.float32), self.state,
                                                   torch.zeros(1), True)
        a = actions.numpy()[0]
        self.prev_action = a.astype(np.float32)
        sy = a[1] * 0.5 - 1.0
        return np.array([a[0] - 1.0, sy, a[2] * 0.5 - 1.0, sy, a[3] - 1.0, a[4], a[5], a[6]], dtype=np.float64)
