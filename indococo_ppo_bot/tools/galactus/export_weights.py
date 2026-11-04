import sys, json, struct, numpy as np
sys.path.insert(0, "/home/ubuntu/ippo_runs/code_fr9/indococo_ppo_bot/tools/pool_eval")
from poolkit.botpack5 import load_gigalearn_seq
out = {}
blob = bytearray(); meta = []
for part in ("SHARED_HEAD", "POLICY"):
    seq = load_gigalearn_seq(f"/home/ubuntu/eval_bots/galactus/rlbot/{part}.lt", "leaky_relu")
    print(part, seq)
    for i, m in enumerate(seq):
        for k, v in m.state_dict().items():
            a = v.detach().numpy().astype("<f4")
            meta.append({"name": f"{part}.{i}.{k}", "kind": type(m).__name__, "shape": list(a.shape), "offset": len(blob)//4})
            blob += a.tobytes()
open("galactus_weights.bin", "wb").write(blob)
json.dump(meta, open("galactus_weights.json", "w"), indent=0)
