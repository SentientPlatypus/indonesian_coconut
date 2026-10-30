"""Function-preserving network growth for an rlgym_ppo checkpoint (Net2Net style).

    python tools/expand_net.py <src ckpt dir> <dst ckpt dir> --layers 3072,3072,3072,1536,1536,1536

The new hidden sizes must contain the old ones as an ordered subsequence of
"widened" layers (new size >= old size); every other new layer must be square
(same size as the layer before it) and is initialised as the identity, which is
exact after a ReLU because its input is non-negative. Widened units get small
random incoming weights and zero outgoing weights, so policy and value outputs
are unchanged. Adam moments are carried over for the old entries and zero for
the new ones.
"""
import argparse
import os
import shutil

import torch

MODEL_FILES = {"PPO_POLICY.pt": "PPO_POLICY_OPTIMIZER.pt", "PPO_VALUE_NET.pt": "PPO_VALUE_NET_OPTIMIZER.pt"}


def linear_layers(sd):
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("model.") and k.endswith(".weight")})
    return [(sd[f"model.{i}.weight"], sd[f"model.{i}.bias"]) for i in idx]


def plan(old_hidden, new_hidden):
    """For each new hidden layer: index of the old layer it widens, or None (identity)."""
    out, j = [], 0
    for h in new_hidden:
        if j < len(old_hidden) and h >= old_hidden[j] and (len(new_hidden) - len(out)) >= (len(old_hidden) - j):
            remaining_new = len(new_hidden) - len(out) - 1
            remaining_old = len(old_hidden) - j - 1
            if remaining_new >= remaining_old:
                out.append(j)
                j += 1
                continue
        out.append(None)
    assert j == len(old_hidden), f"cannot map {old_hidden} into {new_hidden}"
    for k, src in enumerate(out):
        if src is None:
            assert k > 0 and new_hidden[k] == new_hidden[k - 1], f"identity layer {k} must be square"
    return out


def grow(sd, new_hidden, init_scale, gen):
    layers = linear_layers(sd)
    old_hidden = [w.shape[0] for w, _ in layers[:-1]]
    n_in = layers[0][0].shape[1]
    mapping = plan(old_hidden, new_hidden)
    new_layers, maps = [], []   # maps: per new layer, (old weight idx or None, in_old, out_old)
    prev_new, prev_old = n_in, n_in
    for k, h in enumerate(new_hidden):
        src = mapping[k]
        w = torch.zeros(h, prev_new)
        b = torch.zeros(h)
        if src is None:
            w[:, :] = torch.eye(h)
            out_old = prev_old
            maps.append((None, prev_old, prev_old))
        else:
            ow, ob = layers[src]
            out_old = ow.shape[0]
            std = init_scale * (2.0 / prev_new) ** 0.5
            w.normal_(0.0, std, generator=gen)
            w[:out_old, :] = 0.0
            w[:out_old, :prev_old] = ow
            b[:out_old] = ob
            maps.append((src, prev_old, out_old))
        new_layers.append((w, b))
        prev_new, prev_old = h, out_old
    ow, ob = layers[-1]
    w = torch.zeros(ow.shape[0], prev_new)
    w[:, :prev_old] = ow
    new_layers.append((w, ob.clone()))
    maps.append((len(layers) - 1, prev_old, ow.shape[0]))
    out = {}
    for i, (w, b) in enumerate(new_layers):
        out[f"model.{2 * i}.weight"] = w
        out[f"model.{2 * i}.bias"] = b
    return out, maps, old_hidden


def grow_optimizer(opt_sd, maps, new_sd):
    """Rebuild Adam state in new parameter order (weight, bias per layer)."""
    old_state = opt_sd["state"]
    step = next(iter(old_state.values()))["step"]
    new_state = {}
    names = sorted(new_sd, key=lambda k: (int(k.split(".")[1]), k.endswith(".bias")))
    for pi, name in enumerate(names):
        li = int(name.split(".")[1]) // 2
        is_bias = name.endswith(".bias")
        src, in_old, out_old = maps[li]
        st = {"step": step.clone() if torch.is_tensor(step) else step,
              "exp_avg": torch.zeros_like(new_sd[name]), "exp_avg_sq": torch.zeros_like(new_sd[name])}
        if src is not None:
            old = old_state[2 * src + int(is_bias)]
            for key in ("exp_avg", "exp_avg_sq"):
                if is_bias:
                    st[key][:out_old] = old[key]
                else:
                    st[key][:out_old, :in_old] = old[key]
        new_state[pi] = st
    groups = opt_sd["param_groups"]
    assert len(groups) == 1
    groups[0]["params"] = list(range(len(names)))
    return {"state": new_state, "param_groups": groups}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--layers", required=True)
    p.add_argument("--init-scale", type=float, default=0.5)
    a = p.parse_args()
    new_hidden = [int(x) for x in a.layers.split(",")]
    gen = torch.Generator().manual_seed(0)
    shutil.copytree(a.src, a.dst)
    for model_f, opt_f in MODEL_FILES.items():
        sd = torch.load(os.path.join(a.src, model_f), map_location="cpu")
        new_sd, maps, old_hidden = grow(sd, new_hidden, a.init_scale, gen)
        torch.save(new_sd, os.path.join(a.dst, model_f))
        opt = torch.load(os.path.join(a.src, opt_f), map_location="cpu")
        torch.save(grow_optimizer(opt, maps, new_sd), os.path.join(a.dst, opt_f))
        print(model_f, old_hidden, "->", new_hidden)


if __name__ == "__main__":
    main()
