"""Warm-start a team-mode actor from a 1v1 actor by widening its input layer.

    python widen_1v1.py --actor actor_1v1.safetensors --team-size 2 --out actor_2v2.pt
    INDOCOCO_TEAM_SIZE=2 indococo-train import actor_2v2.pt <ck_dir>

DefaultObs layout (obs.rs): ball 9, pads 34, own flags 9, own car 20 (= 72),
then 20 per other car, allies first, then opponents. The 1v1 opponent
block (cols 72..92) is copied onto the FIRST opponent's block; ally blocks
and extra opponent blocks start at zero weight, so at step 0 the 2v2 policy
acts exactly like the 1v1 policy against its nearest-in-order opponent and
learns to use teammates from there.

An FR8 actor (extended obs, 92 + 14) keeps its 14 extension columns
(previous action, car-frame ball terms) at the end of the team obs.
"""
import argparse
import json
import struct
from collections import OrderedDict

import numpy as np
import torch

CAR = 20
HEAD = 72  # ball 9 + pads 34 + own flags 9 + own car 20
EXT = 14   # FR8 extension (src/obs.rs ext_features), after all car blocks


def load_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        data = f.read()
    return {k: torch.from_numpy(np.frombuffer(data[m["data_offsets"][0]:m["data_offsets"][1]], np.float32)
                                .reshape(m["shape"]).copy())
            for k, m in h.items() if k != "__metadata__"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--team-size", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = load_st(a.actor)
    first = min((k for k in sd if k.endswith("weight")), key=lambda k: int(k.split(".")[1]))
    w = sd[first]
    ext = EXT if w.shape[1] == HEAD + CAR + EXT else 0
    assert w.shape[1] == HEAD + CAR + ext, f"{first} {tuple(w.shape)} is not a 1v1 input layer"
    n = a.team_size
    obs = HEAD + CAR * (2 * n - 1) + ext
    nw = torch.zeros(w.shape[0], obs)
    if ext:
        nw[:, obs - EXT:] = w[:, HEAD + CAR:]
    nw[:, :HEAD] = w[:, :HEAD]
    opp0 = HEAD + CAR * (n - 1)
    nw[:, opp0:opp0 + CAR] = w[:, HEAD:HEAD + CAR]
    sd[first] = nw
    torch.save(OrderedDict((k, v.contiguous()) for k, v in sd.items()), a.out)
    print(json.dumps({"layer": first, "in": obs, "opp_block": [opp0, opp0 + CAR], "out": a.out}))


if __name__ == "__main__":
    main()
