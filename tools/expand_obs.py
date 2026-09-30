"""Transfer a 1v1 checkpoint to NvN by widening the first layer.

DefaultObs(zero_padding=None) = [52 shared][self 20][allies 20*(N-1)][enemies 20*N].
The 1v1 layout is [52][self][enemy]. Old columns are copied to the same
features; the new ally / extra-enemy columns start at 0, so the widened net
initially plays exactly like the 1v1 policy (ignoring the new cars).
Adam moments for the widened weight are padded with 0.

    python tools/expand_obs.py <src_ckpt_dir> <dst_ckpt_dir> --team-size 2
"""
import argparse
import json
import os
import shutil

import torch

SHARED, CAR = 52, 20


def widen(w, n):
    """w: (out, 92) -> (out, 52 + 20*2n)."""
    out = w.new_zeros((w.shape[0], SHARED + CAR * 2 * n))
    out[:, :SHARED + CAR] = w[:, :SHARED + CAR]                      # shared + self
    e0 = SHARED + CAR + CAR * (n - 1)                                 # first enemy slot
    out[:, e0:e0 + CAR] = w[:, SHARED + CAR:SHARED + 2 * CAR]
    return out


def first_weight_key(sd):
    return next(k for k in sd if k.endswith(".weight"))


def expand_net(path_in, path_out, n):
    sd = torch.load(path_in, map_location="cpu")
    k = first_weight_key(sd)
    assert sd[k].shape[1] == SHARED + 2 * CAR, f"{path_in}: not a 1v1 net ({sd[k].shape})"
    sd[k] = widen(sd[k], n)
    torch.save(sd, path_out)
    return k


def expand_opt(path_in, path_out, n):
    sd = torch.load(path_in, map_location="cpu")
    for st in sd["state"].values():
        for m in ("exp_avg", "exp_avg_sq"):
            t = st.get(m)
            if t is not None and t.dim() == 2 and t.shape[1] == SHARED + 2 * CAR:
                st[m] = widen(t, n)
    torch.save(sd, path_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--team-size", type=int, required=True)
    a = p.parse_args()
    n = a.team_size
    os.makedirs(a.dst, exist_ok=True)
    for f in ("PPO_POLICY.pt", "PPO_VALUE_NET.pt"):
        expand_net(os.path.join(a.src, f), os.path.join(a.dst, f), n)
    for f in ("PPO_POLICY_OPTIMIZER.pt", "PPO_VALUE_NET_OPTIMIZER.pt"):
        expand_opt(os.path.join(a.src, f), os.path.join(a.dst, f), n)
    bk = json.load(open(os.path.join(a.src, "BOOK_KEEPING_VARS.json")))
    bk["cumulative_timesteps"] = 0
    bk.pop("wandb_run_id", None)
    json.dump(bk, open(os.path.join(a.dst, "BOOK_KEEPING_VARS.json"), "w"), indent=4)
    print(f"expanded {a.src} -> {a.dst} for {n}v{n}: inputs {SHARED + 2 * CAR} -> {SHARED + CAR * 2 * n}")


if __name__ == "__main__":
    main()
