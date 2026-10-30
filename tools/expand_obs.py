"""Transfer a checkpoint between team sizes by remapping the first layer.

DefaultObs(zero_padding=None) = [52 shared][self 20][allies 20*(N-1)][enemies 20*N].
Shared, self, and every ally / enemy slot present in both sizes are copied to the
same features. Slots only in the target start at 0 (so 1v1 -> NvN initially plays
exactly like the 1v1 policy); slots only in the source are dropped (e.g. 3v3 -> 2v2).
Adam moments for the remapped weight get the same treatment.

    python tools/expand_obs.py <src_ckpt_dir> <dst_ckpt_dir> --team-size 2
"""
import argparse
import json
import os
import shutil

import torch

SHARED, CAR = 52, 20


def team_size_of(width):
    n, r = divmod(width - SHARED, 2 * CAR)
    assert r == 0 and n >= 1, f"unexpected obs width {width}"
    return n


def widen(w, n):
    """w: (out, 52 + 40m) -> (out, 52 + 40n)."""
    m = team_size_of(w.shape[1])
    out = w.new_zeros((w.shape[0], SHARED + CAR * 2 * n))
    out[:, :SHARED + CAR] = w[:, :SHARED + CAR]                      # shared + self
    for i in range(min(m, n) - 1):                                    # allies
        out[:, SHARED + CAR * (1 + i):SHARED + CAR * (2 + i)] = w[:, SHARED + CAR * (1 + i):SHARED + CAR * (2 + i)]
    e_src, e_dst = SHARED + CAR * m, SHARED + CAR * n                 # first enemy slot
    for i in range(min(m, n)):
        out[:, e_dst + CAR * i:e_dst + CAR * (i + 1)] = w[:, e_src + CAR * i:e_src + CAR * (i + 1)]
    return out


def first_weight_key(sd):
    return next(k for k in sd if k.endswith(".weight"))


def expand_net(path_in, path_out, n):
    sd = torch.load(path_in, map_location="cpu")
    k = first_weight_key(sd)
    m = team_size_of(sd[k].shape[1])
    sd[k] = widen(sd[k], n)
    torch.save(sd, path_out)
    return m


def expand_opt(path_in, path_out, n):
    sd = torch.load(path_in, map_location="cpu")
    for st in sd["state"].values():
        for m in ("exp_avg", "exp_avg_sq"):
            t = st.get(m)
            if t is not None and t.dim() == 2 and (t.shape[1] - SHARED) % (2 * CAR) == 0 and t.shape[1] > SHARED:
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
        m = expand_net(os.path.join(a.src, f), os.path.join(a.dst, f), n)
    for f in ("PPO_POLICY_OPTIMIZER.pt", "PPO_VALUE_NET_OPTIMIZER.pt"):
        expand_opt(os.path.join(a.src, f), os.path.join(a.dst, f), n)
    bk = json.load(open(os.path.join(a.src, "BOOK_KEEPING_VARS.json")))
    bk["cumulative_timesteps"] = 0
    bk.pop("wandb_run_id", None)
    json.dump(bk, open(os.path.join(a.dst, "BOOK_KEEPING_VARS.json"), "w"), indent=4)
    print(f"remapped {a.src} -> {a.dst}: {m}v{m} -> {n}v{n}, inputs {SHARED + CAR * 2 * m} -> {SHARED + CAR * 2 * n}")


if __name__ == "__main__":
    main()
