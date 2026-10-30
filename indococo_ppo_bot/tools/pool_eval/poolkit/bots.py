"""Adapters that run public RLBot bots headless, using their own unmodified code.

Each adapter loads the bot's model and obs/sequence modules from its directory
(see `fetch_bots.sh`) and replays the decision logic of its RLBot `bot.py`:
which players go into the obs, the previous-action input, the action parser,
the tick skip and any hard-coded kickoff. Differences from running in RLBot:

- Actions apply on the tick they are computed, as in RLGym training. RLBot runs
  them a few ticks late.
- The kickoff phase is "the ball has not moved since the reset". There is no
  countdown.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
from .agent import Agent
from .v1compat import V1PlayerData, install_stub_modules, players_for, to_v1
from .view import AgentView


def load_module(path: str, name: str):
    """Imports `path` as module `name`, with its directory on sys.path for sibling imports."""
    install_stub_modules()
    directory = os.path.dirname(path)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _controls(c) -> list[float]:
    return [c["throttle"], c.get("steer", 0), c.get("pitch", 0), c.get("yaw", 0),
            c.get("roll", 0), c.get("jump", 0), c.get("boost", 0), c.get("handbrake", 0)]


# Nexto's and Necto's hard-coded speedflip kickoff (bot.py KICKOFF_CONTROLS), one row per tick.
NECTO_KICKOFF = np.array(
    11 * 4 * [_controls(dict(throttle=1, boost=1))]
    + 4 * 4 * [_controls(dict(throttle=1, boost=1, steer=-1))]
    + 2 * 4 * [_controls(dict(throttle=1, jump=1, boost=1))]
    + 1 * 4 * [_controls(dict(throttle=1, boost=1))]
    + 1 * 4 * [_controls(dict(throttle=1, yaw=0.8, pitch=-0.7, jump=1, boost=1))]
    + 13 * 4 * [_controls(dict(throttle=1, pitch=1, boost=1))]
    + 10 * 4 * [_controls(dict(throttle=1, roll=1, pitch=0.5))],
    dtype=np.float64,
)


class _V1Bot(Agent):
    """Common flow: v1 state -> bot obs -> bot action; previous action fed back."""

    sort_by_ball = False  # Element / Immortal pick one opponent by ball distance

    def reset(self, view: AgentView) -> None:
        super().reset(view)
        self.action = np.zeros(8)

    def v1(self, view: AgentView):
        gs, me = to_v1(view, self.tick_skip)
        gs.players = players_for(gs, me, self.sort_by_ball)
        return gs, me


class _NectoFamily(_V1Bot):
    """Nexto and Necto: attention model over all entities, 90-row table / 5-way factored action."""

    def __init__(self, directory: str, agent_file: str, obs_file: str, obs_class: str) -> None:
        super().__init__()
        tag = os.path.basename(os.path.normpath(directory))
        self.agent = load_module(os.path.join(directory, agent_file), f"{tag}_agent").Agent()
        self._obs_module = load_module(os.path.join(directory, obs_file), f"{tag}_obs")
        self._obs_class = obs_class
        self.kickoff_index = -1

    def make_obs_builder(self):
        raise NotImplementedError

    def reset(self, view: AgentView) -> None:
        super().reset(view)
        self.obs_builder = self.make_obs_builder()
        self.kickoff_index = -1
        self._reset_obs = True

    def decide(self, view: AgentView) -> np.ndarray:
        gs, me = self.v1(view)
        if self._reset_obs and hasattr(self.obs_builder, "reset"):
            self.obs_builder.reset(gs)
            self._reset_obs = False
        obs = self.obs_builder.build_obs(me, gs, self.action)
        beta = 0.5 if (self.stochastic_kickoffs and view.kickoff) else 1
        self.action, _ = self.agent.act(obs, beta)
        return self.action

    def controls(self, view: AgentView, tick: int) -> np.ndarray:
        held = super().controls(view, tick)
        if not view.kickoff:
            self.kickoff_index = -1
            return held
        # 1v1: always the kickoff taker. Index advances by one per tick.
        self.kickoff_index = 0 if self.kickoff_index < 0 else self.kickoff_index + 1
        if self.kickoff_index < len(NECTO_KICKOFF):
            self.action = NECTO_KICKOFF[self.kickoff_index]
            self._held = self.action
        return self._held

    def ticks_until_next_call(self, view: AgentView, tick: int) -> int:
        if view.kickoff and self.kickoff_index + 1 < len(NECTO_KICKOFF):
            return 1
        return super().ticks_until_next_call(view, tick)


class Nexto(_NectoFamily):
    stochastic_kickoffs = True

    def __init__(self, directory: str) -> None:
        super().__init__(directory, "agent.py", "nexto_obs.py", "NextoObsBuilder")

    def make_obs_builder(self):
        return self._obs_module.NextoObsBuilder(field_info=None)


class Necto(_NectoFamily):
    stochastic_kickoffs = False

    def __init__(self, directory: str) -> None:
        super().__init__(directory, "agent.py", "necto_obs.py", "NectoObsBuilder")

    def make_obs_builder(self):
        # Necto was trained with pads in RLGym's BOOST_LOCATIONS order and heights 70/73.
        from types import SimpleNamespace

        locations = self._nexto_boost_locations()
        pads = [
            SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=z), is_full_boost=z > 72)
            for x, y, z in locations
        ]
        return self._obs_module.NectoObsBuilder(SimpleNamespace(boost_pads=pads))

    @staticmethod
    def _nexto_boost_locations():
        from rlgym.rocket_league.common_values import BOOST_LOCATIONS

        return [(x, y, 73.0 if z > 71 else 70.0) for x, y, z in BOOST_LOCATIONS]


class Element(_V1Bot):
    """Element: 5 categorical (3-way) + 3 Bernoulli heads, speedflip kickoff, yaw 0 while jumping."""

    sort_by_ball = True
    action_trans = np.array([-1, -1, -1, -1, -1, 0, 0, 0])

    def __init__(self, directory: str) -> None:
        super().__init__()
        agent_mod = load_module(os.path.join(directory, "agent.py"), "element_agent")
        obs_mod = load_module(os.path.join(directory, "obs.py"), "element_obs")
        # The bot's own `sequences` package (relative imports), importable once its dir is on sys.path.
        self._speedflip = importlib.import_module("sequences.speedflip").Speedflip
        self.obs_builder = obs_mod.CustomObs(cars=2)
        self.agent = agent_mod.Agent(self.obs_builder.obs_size, action_categoricals=5, action_bernoullis=3)

    def reset(self, view: AgentView) -> None:
        super().reset(view)
        self.kickoff_seq = None

    def decide(self, view: AgentView) -> np.ndarray:
        gs, me = self.v1(view)
        obs = self.obs_builder.build_obs(me, gs, self.action)
        self.action = self.agent.act(obs)[0] + self.action_trans
        return self._apply_quirks(self.action)

    @staticmethod
    def _apply_quirks(action: np.ndarray) -> np.ndarray:
        out = np.array(action, dtype=np.float64)
        if out[5] > 0:
            out[3] = 0  # update_controls: yaw = 0 while jumping
        return out

    def _kickoff_action(self, view: AgentView):
        if not view.kickoff:
            self.kickoff_seq = None
            return None
        gs, me = to_v1(view, self.tick_skip)
        if self.kickoff_seq is None:
            self.kickoff_seq = self._speedflip(me)
        if not self.kickoff_seq.is_valid(me, gs):
            return None
        self.action = np.asarray(self.kickoff_seq.get_action(me, gs, self.action), dtype=np.float64)
        return self._apply_quirks(self.action)

    def controls(self, view: AgentView, tick: int) -> np.ndarray:
        kick = self._kickoff_action(view)
        if kick is not None:
            # RLBot's Element returns before the model step during its kickoff.
            return kick
        return super().controls(view, tick)

    def ticks_until_next_call(self, view: AgentView, tick: int) -> int:
        if view.kickoff and (self.kickoff_seq is None or self.kickoff_seq.state != "done"):
            return 1
        return super().ticks_until_next_call(view, tick)


class Immortal(_V1Bot):
    """Immortal: AdvancedObs + its 90-row table (handbrake on all aerial rows), tick skip 6."""

    sort_by_ball = True
    tick_skip = 6

    def __init__(self, directory: str) -> None:
        super().__init__()
        install_stub_modules()
        sys.path.insert(0, directory)
        agent_mod = load_module(os.path.join(directory, "agent.py"), "immortal_agent")
        obs_mod = load_module(os.path.join(directory, "obs", "advanced_obs.py"), "immortal_obs")
        self.agent = agent_mod.Agent()
        self.parser = agent_mod.ImmortalAction()
        self.obs_builder = obs_mod.ExpandAdvancedObs()

    def decide(self, view: AgentView) -> np.ndarray:
        gs, me = self.v1(view)
        obs = self.obs_builder.build_obs(me, gs, self.action)
        self.action = np.asarray(self.parser.parse_actions(self.agent.act(obs)), dtype=np.float64)
        return Element._apply_quirks(self.action)

