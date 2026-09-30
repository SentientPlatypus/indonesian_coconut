import os
from collections import OrderedDict

import numpy as np
import torch
from rlbot.flat import ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot
from rlgym_compat import GameState, common_values

from act import LookupTableAction
from custom_discrete import DiscreteFF
from obs import DefaultObs
from rlgym_compat.sim_extra_info import SimExtraInfo

HERE = os.path.dirname(os.path.abspath(__file__))
# One policy per team size; each was trained on DefaultObs with exactly that
# many cars per team (obs = 52 + 20 * 2N inputs).
POLICY_FILES = {1: "policies/1v1.pt", 2: "policies/2v2.pt", 3: "policies/3v3.pt"}
TICK_SKIP = 8


def model_info_from_dict(loaded_dict):
    state_dict = OrderedDict(loaded_dict)
    bias_counts, weight_counts = [], []
    for key, value in state_dict.items():
        if ".weight" in key:
            weight_counts.append(value.numel())
        if ".bias" in key:
            bias_counts.append(value.size(0))
    inputs = int(weight_counts[0] / bias_counts[0])
    return inputs, bias_counts[-1], bias_counts[:-1]


def make_obs(team_size):
    return DefaultObs(
        zero_padding=team_size,
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y,
                             1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0)


class IndonesianCoconut(Bot):

    def initialize(self):
        self.deterministic = False
        self.ticks = self.tick_skip = TICK_SKIP
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        self.policies = {}
        self.obs_builders = {}

        self.game_state = GameState()
        self.prev_time = 0.0
        self.prev_control = ControllerState()
        self.controls = ControllerState()
        self.action_parser = LookupTableAction()
        self.extra_info = SimExtraInfo(self.field_info, tick_skip=self.tick_skip)
        self.game_state = self.game_state.create_compat_game_state(self.field_info, tick_skip=self.tick_skip)
        # load all three up front so switching mode mid-series never stalls a tick
        for n in POLICY_FILES:
            self._policy(n)

    def _policy(self, team_size):
        if team_size not in self.policies:
            sd = torch.load(os.path.join(HERE, POLICY_FILES[team_size]), map_location=self.device)
            inputs, n_actions, layers = model_info_from_dict(sd)
            assert inputs == 52 + 40 * team_size, f"{POLICY_FILES[team_size]}: {inputs} inputs"
            pol = DiscreteFF(inputs, n_actions, layers, self.device)
            pol.load_state_dict(sd)
            pol.eval()
            self.policies[team_size] = pol
            self.obs_builders[team_size] = make_obs(team_size)
        return self.policies[team_size]

    def _team_size(self, packet: GamePacket) -> int:
        blue = sum(1 for p in packet.players if p.team == 0)
        orange = sum(1 for p in packet.players if p.team == 1)
        return int(min(3, max(1, blue, orange)))

    def get_output(self, packet: GamePacket) -> ControllerState:
        cur_time = packet.match_info.frame_num
        ticks_elapsed = cur_time - self.prev_time
        self.prev_time = cur_time
        self.ticks += ticks_elapsed

        if len(packet.balls) == 0 or packet.match_info.match_phase == MatchPhase.Ended:
            return ControllerState()

        extra_info = self.extra_info.get_extra_info(packet)
        self.game_state.update(packet, extra_info=extra_info)

        if self.ticks < self.tick_skip - 1:
            return self.prev_control
        self.ticks = 0

        n = self._team_size(packet)
        policy = self._policy(n)
        obs = self.obs_builders[n].build_obs(list(self.game_state.cars.keys()), self.game_state, {})
        obs = np.asarray(obs.get(self.player_id), dtype=np.float32).flatten()
        with torch.no_grad():
            action_idx, _ = policy.get_action(torch.as_tensor(obs, device=self.device),
                                              deterministic=self.deterministic)
            if self.deterministic:
                action_idx = torch.tensor([action_idx], device=self.device)

        parsed = self.action_parser.parse_actions(
            actions={self.player_id: action_idx}, state=self.game_state, shared_info={}).get(self.player_id)
        if len(parsed.shape) == 2 and parsed.shape[0] == 1:
            parsed = parsed[0]
        try:
            self.update_controls(parsed)
        except Exception:
            return self.prev_control
        self.prev_control = self.controls
        return self.controls

    def update_controls(self, action):
        a = [float(x) for x in action]
        self.controls.throttle = a[0]
        self.controls.steer = a[1]
        self.controls.pitch = a[2]
        self.controls.yaw = a[3]
        self.controls.roll = a[4]
        self.controls.jump = a[5] > 0
        self.controls.boost = a[6] > 0
        self.controls.handbrake = a[7] > 0


if __name__ == "__main__":
    IndonesianCoconut("indonesian_coconut/indonesian_coconut").run()
