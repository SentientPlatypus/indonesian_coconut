#!/usr/bin/env python3
"""FR5 behavior cloning: fine-tune an exported actor (.safetensors, PyTorch
layout model.{0,2,4,6,8}) on scripted flip reset expert labels, while
distilling the frozen original on normal-play observations so 1v1 play is kept.

    python bc_pretrain.py --actor final_policy.safetensors --curr data/curr_L0 [data/curr_L1 ...] \
        --normal data/normal.f32 --out bc.pt [--steps 4000] [--lr 2e-5] [--kl 1.0]

Loss = CE(student, expert action) on curriculum states
     + kl * KL(teacher || student) on normal states.
Writes a plain state_dict (.pt) that `indococo-train import` reads, and
bc_log.jsonl next to it.
"""
import argparse, json, os, struct, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

OBS, ACT, LAYERS = 92, 90, [2048, 2048, 1024, 1024]


def load_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); data = f.read()
    return {k: torch.from_numpy(np.frombuffer(data[m["data_offsets"][0]:m["data_offsets"][1]], np.float32).reshape(m["shape"]).copy())
            for k, m in h.items() if k != "__metadata__"}


def net():
    mods, d = [], OBS
    for w in LAYERS:
        mods += [nn.Linear(d, w), nn.ReLU()]; d = w
    mods.append(nn.Linear(d, ACT))
    m = nn.Sequential(*mods)
    return nn.ModuleDict({"model": m})


def fwd(m, x):
    return m["model"](x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True)
    ap.add_argument("--curr", nargs="+", required=True)
    ap.add_argument("--normal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--use-weight", type=float, default=1.0, help="weight of use-phase labels")
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    sd = load_st(a.actor)
    student, teacher = net(), net()
    student.load_state_dict(sd); teacher.load_state_dict(sd)
    student.to(dev); teacher.to(dev).eval()
    xs, ys, ws = [], [], []
    for p in a.curr:
        x = np.fromfile(p + ".obs.f32", np.float32).reshape(-1, OBS)
        y = np.fromfile(p + ".act.u8", np.uint8).astype(np.int64)
        ph = np.fromfile(p + ".phase.u8", np.uint8)
        xs.append(x); ys.append(y); ws.append(np.where(ph == 1, a.use_weight, 1.0).astype(np.float32))
    cx = torch.from_numpy(np.concatenate(xs)).to(dev); cy = torch.from_numpy(np.concatenate(ys)).to(dev)
    cw = torch.from_numpy(np.concatenate(ws)).to(dev)
    nx = torch.from_numpy(np.fromfile(a.normal, np.float32).reshape(-1, OBS)).to(dev)
    print(f"curriculum {len(cx)} labelled states, normal {len(nx)} states", flush=True)
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    log = open(os.path.splitext(a.out)[0] + "_log.jsonl", "w")
    t0 = time.time()
    for step in range(a.steps + 1):
        i = torch.randint(len(cx), (a.batch,), device=dev); j = torch.randint(len(nx), (a.batch,), device=dev)
        lc = fwd(student, cx[i])
        ce = (F.cross_entropy(lc, cy[i], reduction="none") * cw[i]).sum() / cw[i].sum()
        acc = (lc.argmax(1) == cy[i]).float().mean()
        with torch.no_grad():
            tl = F.log_softmax(fwd(teacher, nx[j]), 1)
        sl = F.log_softmax(fwd(student, nx[j]), 1)
        kl = (tl.exp() * (tl - sl)).sum(1).mean()
        loss = ce + a.kl * kl
        if step < a.steps:
            opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                agree = (sl.argmax(1) == tl.argmax(1)).float().mean()
            r = dict(step=step, ce=ce.item(), acc=acc.item(), kl=kl.item(), normal_argmax_agree=agree.item(), s=round(time.time() - t0, 1))
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
    out = {k: v.detach().cpu().contiguous() for k, v in student.state_dict().items()}
    from collections import OrderedDict
    torch.save(OrderedDict(out), a.out)
    print("wrote", a.out, list(out)[:3])


if __name__ == "__main__":
    main()
