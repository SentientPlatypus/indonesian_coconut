"""FR8 extended obs: 14 floats after the base DefaultObs (src/obs.rs `ext_features`).

[0:8]  previous action = the 8 held controls (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
[8:11] ball position minus own car position in the car frame (forward, forward x up, up) / 4096
[11:14] ball velocity minus own car velocity in the same frame / 2300
The car-frame terms are computed from the base obs (own car at 52..72), exactly as in Rust.
"""
from __future__ import annotations

import numpy as np

EXT_LEN = 14
POS_COEF = np.array([1 / 4096.0, 1 / 6000.0, 1 / 2044.0], np.float32)
LIN_VEL_COEF = np.float32(1 / 2300.0)
REL_POS_COEF = np.float32(1 / 4096.0)


def is_ext_width(n: int) -> bool:
    """True for an extended-obs input width (106, 146, 186)."""
    return n > 52 and (n - 52) % 40 == EXT_LEN


def base_width(n: int) -> int:
    return n - EXT_LEN if is_ext_width(n) else n


def ext_features(base: np.ndarray, prev) -> np.ndarray:
    """`base`: [..., >=72] base obs rows; `prev`: [..., 8] held controls. Returns [..., 14] float32."""
    base = np.asarray(base, np.float32)
    prev = np.broadcast_to(np.asarray(prev, np.float32), base.shape[:-1] + (8,))
    dp = base[..., 0:3] / POS_COEF - base[..., 52:55] / POS_COEF
    dv = (base[..., 3:6] - base[..., 61:64]) / LIN_VEL_COEF
    fwd, up = base[..., 55:58], base[..., 58:61]
    side = np.cross(fwd, up)
    out = [prev]
    for d, s in ((dp, REL_POS_COEF), (dv, LIN_VEL_COEF)):
        out.append(np.stack([(d * fwd).sum(-1), (d * side).sum(-1), (d * up).sum(-1)], -1) * s)
    return np.concatenate(out, -1).astype(np.float32)


def extend(base: np.ndarray, prev) -> np.ndarray:
    return np.concatenate([np.asarray(base, np.float32), ext_features(base, prev)], -1)
