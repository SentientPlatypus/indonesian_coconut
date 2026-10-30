"""Our policy: the 90-action DiscreteFF used by 1v1.pt and the Rust bot.

Observations are RLGym v2 `DefaultObs` with the `v4_env` coefficients and,
by default, the 1.25 s `air_time_since_jump` clamp. The clamp is what the
deployed bot sees (rlgym_compat freezes the timer at 1.25 s) and what the Rust
`V2DefaultObs` trains on. `h2h_1v1.py` on master builds its env with
`clamp_air_time` off; pass `clamp_air_time = false` to reproduce it. Weights load from a `.pt` state dict
(`model.{0,2,4,...}`), an exported `.safetensors`, or a Rust checkpoint
directory (exported on the fly with `indococo-train export`).
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile

import numpy as np
import torch
from torch import nn

from .agent import Agent
from .view import AgentView

AIR_TIME_INDEX = 9 + 34 + 8
AIR_TIME_CLAMP = 1.25


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    dtypes = {"F32": np.float32, "F16": np.float16, "F64": np.float64}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        arr = np.frombuffer(data[a:b], dtype=dtypes[meta["dtype"]]).reshape(meta["shape"])
        out[key] = torch.from_numpy(arr.astype(np.float32))
    return out


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        train_bin = os.environ.get("TRAIN_BIN", "indococo-train")
        out = os.path.join(tempfile.mkdtemp(prefix="pool_eval_"), "policy.safetensors")
        subprocess.run([train_bin, "export", path, out], check=True, stdout=subprocess.DEVNULL)
        return _load_safetensors(out)
    if path.endswith(".safetensors"):
        return _load_safetensors(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def build_mlp(state_dict: dict[str, torch.Tensor]) -> nn.Sequential:
    """Rebuild `Linear, ReLU, ..., Linear` from `model.{i}.weight` keys."""
    linear_ids = sorted({int(k.split(".")[1]) for k in state_dict if k.endswith(".weight")})
    layers: list[nn.Module] = []
    for n, i in enumerate(linear_ids):
        w = state_dict[f"model.{i}.weight"]
        layer = nn.Linear(w.shape[1], w.shape[0])
        layer.weight.data.copy_(w)
        layer.bias.data.copy_(state_dict[f"model.{i}.bias"])
        layers.append(layer)
        if n < len(linear_ids) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers).eval()


def default_obs_builder():
    from rlgym.rocket_league import common_values as cv
    from rlgym.rocket_league.obs_builders import DefaultObs

    return DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1 / cv.SIDE_WALL_X, 1 / cv.BACK_NET_Y, 1 / cv.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / cv.CAR_MAX_SPEED,
        ang_vel_coef=1 / cv.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )


class IndococoPolicy(Agent):
    """Greedy (argmax) DiscreteFF over the RLGym lookup table, 8-tick skip."""

    def __init__(self, path: str, clamp_air_time: bool = True) -> None:
        super().__init__()
        self.clamp_air_time = clamp_air_time
        from rlgym.rocket_league.action_parsers import LookupTableAction

        self.model = build_mlp(load_state_dict(path))
        self.table = LookupTableAction.make_lookup_table()
        self.obs_builder = default_obs_builder()

    def observe(self, view: AgentView) -> np.ndarray:
        agents = list(view.state.cars)
        obs = self.obs_builder.build_obs(agents, view.state, {})[view.car_id]
        obs = np.array(obs, dtype=np.float32)
        if self.clamp_air_time:
            obs[AIR_TIME_INDEX] = min(obs[AIR_TIME_INDEX], AIR_TIME_CLAMP)
        return obs

    @torch.no_grad()
    def decide(self, view: AgentView) -> np.ndarray:
        logits = self.model(torch.from_numpy(self.observe(view)))
        return self.table[int(logits.argmax())]
