"""FR8 extended-obs parity: Python `poolkit.obs_ext` vs the Rust obs on states
recorded by `INDOCOCO_OBS_EXT=1 indococo-train record <rewards.toml> <out.jsonl> <steps> [ckpt]`.

    python obs_ext_parity.py rec.jsonl

For every car of every recorded step, recomputes the 14 extension floats from
the recorded base obs and the car's recorded held controls and compares them
with the Rust ones. Exits non-zero on a mismatch above 1e-4 (relative).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pool_eval"))
from poolkit.obs_ext import EXT_LEN, base_width, ext_features

worst, rows, prev_ok = 0.0, 0, 0
for line in open(sys.argv[1]):
    d = json.loads(line)
    for o, x in zip(d["obs"], d["extra"]):
        o = np.asarray(o, np.float32); b = base_width(len(o))
        assert len(o) == b + EXT_LEN, f"not an extended obs: {len(o)}"
        py = ext_features(o[:b], np.asarray(x["controls"], np.float32))
        err = np.abs(py - o[b:]) / np.maximum(np.abs(o[b:]), 1.0)
        worst = max(worst, float(err.max())); rows += 1
        prev_ok += int(np.array_equal(py[:8], o[b:b + 8]))
print(json.dumps({"rows": rows, "max_rel_err": worst, "prev_action_exact": prev_ok}))
sys.exit(0 if worst < 1e-4 and prev_ok == rows else 1)
