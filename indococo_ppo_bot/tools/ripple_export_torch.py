#!/usr/bin/env python3
"""Export Ripple v1.1 ONNX initializers to safetensors for the LibTorch forward.

The ONNX graph is ``ExportNetwork`` from Ripple-RLBot ``tools/export_network.py``
at 7ca4273a. This script rebuilds that module's state dict in PyTorch layout
(linear weights ``[out, in]``) and writes it next to the ONNX file.

Architecture (both ``ripple-v1.1.onnx`` and ``ripple-v1.1-bcm.onnx``):

* dim 128, 4 heads (head dim 32), 4 cross-attention layers.
* Each MLP is Linear, GELU, Linear, GELU, Linear, GELU, Linear.
  GELU is the erf form ``0.5 * x * (1 + erf(x / sqrt(2)))``.
* ``main`` uses the first 30 floats. ``main[:, 30] > 0.5`` swaps in the
  orange action permutation. ``others[:, :, 15] == 0`` masks a padded player.
* KV is 8 tokens: 5 other players, the ball (6*9 flattened), the action queue
  (6*8 flattened), and the max over the 34 boost-pad embeddings.
* Each layer: LayerNorm Q and KV (eps 1e-5), packed QKV linear, scale by
  ``sqrt(32)``, mask fill ``-inf``, softmax, out projection, residual,
  LayerNorm, MLP (128 -> 512 -> 128), residual.
* Logits are ``LayerNorm(query) @ action_embeddings.T`` (276 actions).

``--check`` runs the rebuilt weights against ONNX Runtime on CPU.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import load_file, save_file

DIM = 128
HEADS = 4
LAYERS = 4
N_ACTIONS = 276

# ONNX MatMul stores W.T as [in, out]. These names are the folded constants.
MATMUL = {
    "other_players_processor": ("val_46", "val_55", "val_64", "val_73"),
    "ball_processor": ("val_80", "val_89", "val_98", "val_107"),
    "action_queue_processor": ("val_114", "val_123", "val_132", "val_141"),
    "boost_processor": ("val_143", "val_152", "val_161", "val_170"),
}
ATTN = (
    ("slice_3", "slice_5", "val_249", "val_253", "val_262"),
    ("slice_7", "slice_9", "val_336", "val_340", "val_349"),
    ("slice_11", "slice_13", "val_423", "val_427", "val_436"),
    ("slice_15", "slice_17", "val_510", "val_514", "val_523"),
)


def initializers(path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(path)
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def linear_weight(weights: dict[str, np.ndarray], prefix: str, index: int, matmul: str | None) -> np.ndarray:
    if matmul is None:
        return np.ascontiguousarray(weights[f"{prefix}.{index}.weight"])
    # [in, out] -> [out, in]
    return np.ascontiguousarray(weights[matmul].T)


def export_state(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for prefix, mats in (
        ("main_player_processor", (None, None, None, None)),
        ("other_players_processor", MATMUL["other_players_processor"]),
        ("ball_processor", MATMUL["ball_processor"]),
        ("action_queue_processor", MATMUL["action_queue_processor"]),
        ("boost_processor", MATMUL["boost_processor"]),
    ):
        for index, matmul in zip((0, 2, 4, 6), mats):
            out[f"{prefix}.{index}.weight"] = linear_weight(weights, prefix, index, matmul)
            out[f"{prefix}.{index}.bias"] = np.ascontiguousarray(weights[f"{prefix}.{index}.bias"])
    for i, (q_bias, kv_bias, out_w, mlp0, mlp2) in enumerate(ATTN):
        p = f"layers.{i}"
        out[f"{p}.attn.in_proj_weight"] = np.ascontiguousarray(weights[f"{p}.attn.in_proj_weight"])
        out[f"{p}.attn.in_proj_bias"] = np.ascontiguousarray(np.concatenate([weights[q_bias], weights[kv_bias]]))
        out[f"{p}.attn.out_proj.weight"] = np.ascontiguousarray(weights[out_w].T)
        out[f"{p}.attn.out_proj.bias"] = np.ascontiguousarray(weights[f"{p}.attn.out_proj.bias"])
        for norm in ("norm_q", "norm_kv", "norm_qkv"):
            out[f"{p}.{norm}.weight"] = np.ascontiguousarray(weights[f"{p}.{norm}.weight"])
            out[f"{p}.{norm}.bias"] = np.ascontiguousarray(weights[f"{p}.{norm}.bias"])
        out[f"{p}.mlp.0.weight"] = np.ascontiguousarray(weights[mlp0].T)
        out[f"{p}.mlp.0.bias"] = np.ascontiguousarray(weights[f"{p}.mlp.0.bias"])
        out[f"{p}.mlp.2.weight"] = np.ascontiguousarray(weights[mlp2].T)
        out[f"{p}.mlp.2.bias"] = np.ascontiguousarray(weights[f"{p}.mlp.2.bias"])
    out["state_norm.weight"] = np.ascontiguousarray(weights["state_norm.weight"])
    out["state_norm.bias"] = np.ascontiguousarray(weights["state_norm.bias"])
    # ONNX `permute` is action_embeddings.T, shape [128, 276].
    out["action_embeddings"] = np.ascontiguousarray(weights["permute"].T)
    out["invert_map"] = np.ascontiguousarray(weights["val_545"].reshape(-1).astype(np.int64))
    return out


def gelu(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return (0.5 * x64 * (1.0 + erf(x64 / math.sqrt(2.0)))).astype(np.float32)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=-1, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x64 - mean) / np.sqrt(var + 1e-5) * weight + bias).astype(np.float32)


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (x.astype(np.float64) @ weight.T.astype(np.float64) + bias).astype(np.float32)


def mlp(x: np.ndarray, state: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    for index in (0, 2, 4, 6):
        x = linear(x, state[f"{prefix}.{index}.weight"], state[f"{prefix}.{index}.bias"])
        if index != 6:
            x = gelu(x)
    return x


def forward(state: dict[str, np.ndarray], main, others, ball, actions, boosts) -> np.ndarray:
    batch = main.shape[0]
    mask = np.concatenate([others[:, :, 15] == 0, np.zeros((batch, 3), dtype=bool)], axis=1)
    query = mlp(main[:, :30], state, "main_player_processor")[:, None, :]
    kv = np.concatenate(
        [
            mlp(others, state, "other_players_processor"),
            mlp(ball.reshape(batch, 1, 54), state, "ball_processor"),
            mlp(actions.reshape(batch, 1, 48), state, "action_queue_processor"),
            mlp(boosts, state, "boost_processor").max(axis=1, keepdims=True),
        ],
        axis=1,
    )
    scale = math.sqrt(DIM // HEADS)
    for i in range(LAYERS):
        p = f"layers.{i}"
        qn = layer_norm(query, state[f"{p}.norm_q.weight"], state[f"{p}.norm_q.bias"])
        kvn = layer_norm(kv, state[f"{p}.norm_kv.weight"], state[f"{p}.norm_kv.bias"])
        weight = state[f"{p}.attn.in_proj_weight"]
        bias = state[f"{p}.attn.in_proj_bias"]
        q_lin = linear(qn, weight[:DIM], bias[:DIM])
        kv_lin = linear(kvn, weight[DIM:], bias[DIM:])
        key, value = np.split(kv_lin, 2, axis=-1)
        q_lin = q_lin.reshape(batch, 1, HEADS, DIM // HEADS).transpose(0, 2, 1, 3)
        key = key.reshape(batch, 8, HEADS, DIM // HEADS).transpose(0, 2, 1, 3)
        value = value.reshape(batch, 8, HEADS, DIM // HEADS).transpose(0, 2, 1, 3)
        scores = np.matmul(q_lin.astype(np.float64) / scale, key.astype(np.float64).transpose(0, 1, 3, 2))
        scores = np.where(mask[:, None, None, :], -np.inf, scores)
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp = np.where(np.isfinite(scores), np.exp(shifted), 0.0)
        attn = np.matmul((exp / exp.sum(axis=-1, keepdims=True)), value.astype(np.float64)).astype(np.float32)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, 1, DIM)
        out = linear(attn, state[f"{p}.attn.out_proj.weight"], state[f"{p}.attn.out_proj.bias"])
        result = query + out
        hidden = layer_norm(result, state[f"{p}.norm_qkv.weight"], state[f"{p}.norm_qkv.bias"])
        hidden = gelu(linear(hidden, state[f"{p}.mlp.0.weight"], state[f"{p}.mlp.0.bias"]))
        hidden = linear(hidden, state[f"{p}.mlp.2.weight"], state[f"{p}.mlp.2.bias"])
        query = result + hidden
    normed = layer_norm(query[:, 0, :], state["state_norm.weight"], state["state_norm.bias"])
    logits = (normed.astype(np.float64) @ state["action_embeddings"].T.astype(np.float64)).astype(np.float32)
    inverted = logits[:, state["invert_map"]]
    return np.where(main[:, 30:31] > 0.5, inverted, logits)


def check(onnx_path: Path, state: dict[str, np.ndarray]) -> None:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for batch in (1, 4, 8):
        main = rng.normal(size=(batch, 32)).astype(np.float32)
        others = rng.normal(size=(batch, 5, 21)).astype(np.float32)
        others[:, :, 15] = rng.choice([-1.0, 0.0, 1.0], size=(batch, 5)).astype(np.float32)
        ball = rng.normal(size=(batch, 6, 9)).astype(np.float32)
        actions = rng.normal(size=(batch, 6, 8)).astype(np.float32)
        boosts = rng.normal(size=(batch, 34, 5)).astype(np.float32)
        main[:, 30] = rng.integers(0, 2, size=batch).astype(np.float32)
        got = forward(state, main, others, ball, actions, boosts)
        expected = session.run(
            None,
            {"main": main, "others": others, "ball": ball, "actions": actions, "boosts": boosts},
        )[0]
        err = float(np.max(np.abs(got - expected)))
        scale = max(1.0, float(np.max(np.abs(expected))))
        rel = err / scale
        agree = float(np.mean(got.argmax(-1) == expected.argmax(-1)))
        worst = max(worst, rel)
        print(
            f"  batch {batch}: max |logit diff| {err:.4e} "
            f"(relative {rel:.4e}, logit scale {scale:.3g}), argmax agreement {agree:.3f}"
        )
        if agree < 1.0:
            raise SystemExit(f"{onnx_path.name}: argmax agreement {agree:.3f}")
    # The bcm net's action embeddings are ~10x the policy's, so its logits sit
    # in the thousands and one fp32 ulp is ~1e-2 absolute. Relative error is
    # the comparison that matches both graphs.
    if worst > 1e-4:
        raise SystemExit(f"{onnx_path.name}: relative error {worst:.4e} exceeds 1e-4")
    print(f"  worst relative {worst:.4e}")


def export_one(onnx_path: Path, dest: Path, do_check: bool) -> None:
    raw = initializers(onnx_path)
    state = export_state(raw)
    save_file(
        state,
        dest,
        metadata={
            "arch": "ripple-export-network",
            "source": onnx_path.name,
            "dim": str(DIM),
            "heads": str(HEADS),
            "layers": str(LAYERS),
            "actions": str(N_ACTIONS),
        },
    )
    nbytes = sum(v.nbytes for v in state.values())
    print(f"wrote {dest} ({nbytes / 1e6:.2f} MB, {len(state)} tensors)")
    if do_check:
        check(onnx_path, load_file(dest))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="directory with ripple-v1.1.onnx and ripple-v1.1-bcm.onnx")
    parser.add_argument("--check", action="store_true", help="compare the export to ONNX Runtime")
    args = parser.parse_args()
    pairs = (
        ("ripple-v1.1.onnx", "ripple-v1.1.safetensors"),
        ("ripple-v1.1-bcm.onnx", "ripple-v1.1-bcm.safetensors"),
    )
    for src, dst in pairs:
        export_one(args.model_dir / src, args.model_dir / dst, args.check)


if __name__ == "__main__":
    main()
