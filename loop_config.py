"""
Config schema + hill-climb knobs for the self-improving loop.

Pure Python (no rlgym / torch) so it can be unit-tested anywhere. The loop
serializes a config dict to JSON; the training env factory (v4_env.build_env)
and the Learner both read from that same dict, so a cycle's config is the single
source of truth shared across the trainer subprocesses.

Hill-climb policy: perturb exactly ONE knob per cycle by ±1 step, clamped to
[min, max]. One knob at a time keeps credit assignment interpretable and the
search conservative — important because the eval signal is noisy.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Baseline config == the hand-tuned V4 starting point (mirrors freestyler_v4.py).
# The loop starts here and climbs from it.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "reward_weights": {
        # scoring
        "goal": 1200.0, "goal_prob": 12.0, "ball_travel": 10.0,
        "vel_ball_to_goal": 3.0, "goal_dist": 6.0,
        # chase / possession
        "speed_to_ball": 1.0, "face_ball": 1.0, "touch": 6.0, "possession": 55.0,
        # energy
        "energy": 3.0, "boost_keep": 5.0, "boost_change": 20.0,
        # air
        "aerial_boost": 4.0, "aerial_distance": 40.0, "in_air": 0.0,
        # mechanics
        "airdribble": 45.0, "airdribble_seq": 30.0, "wall_pop": 8.0,
        "flick": 10.0, "flip_reset": 140.0,
        # hygiene
        "recover": 45.0, "demo": 50.0, "ang_vel": 1.0,
    },
    "airdribble_w_goal_align": 0.25,
    "curriculum": {"kickoff_w": 0.50, "air_dribble_w": 0.30, "flip_reset_w": 0.20},
    "ppo_ent_coef": 0.01,
}

# Tunable knobs the hill-climb is allowed to touch: (dotted_path, min, max, step).
# Deliberately a SMALL, mechanic/scoring-focused subset — not all 23 weights.
# Bounds are sane guardrails so the search can't drive a weight to an absurd value.
TUNABLES: List[Tuple[str, float, float, float]] = [
    ("reward_weights.flip_reset",        80.0, 220.0, 25.0),
    ("reward_weights.airdribble",        25.0,  70.0,  8.0),
    ("reward_weights.airdribble_seq",    15.0,  50.0,  8.0),
    ("reward_weights.wall_pop",           0.0,  24.0,  4.0),
    ("reward_weights.flick",              4.0,  24.0,  4.0),
    ("reward_weights.goal_prob",          8.0,  20.0,  2.0),
    ("reward_weights.possession",        30.0,  75.0,  8.0),
    ("airdribble_w_goal_align",           0.0,   0.6,  0.1),
    ("curriculum.air_dribble_w",          0.15,  0.45, 0.05),
    ("curriculum.flip_reset_w",           0.05,  0.35, 0.05),
    ("ppo_ent_coef",                      0.008, 0.02, 0.002),
]


# ---- nested dotted-path get/set ------------------------------------------
def get_path(cfg: Dict[str, Any], path: str) -> float:
    node: Any = cfg
    for key in path.split("."):
        node = node[key]
    return float(node)


def set_path(cfg: Dict[str, Any], path: str, value: float) -> None:
    keys = path.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_curriculum(cfg: Dict[str, Any]) -> None:
    """Keep curriculum weights >= 0 and re-derive kickoff_w so the three sum to 1.

    The hill-climb only ever moves air_dribble_w / flip_reset_w; kickoff_w is
    whatever is left (floored at 0.20 so we never starve real-game play).
    """
    c = cfg["curriculum"]
    ad = _clamp(float(c.get("air_dribble_w", 0.30)), 0.0, 0.70)
    fr = _clamp(float(c.get("flip_reset_w", 0.20)), 0.0, 0.70)
    if ad + fr > 0.80:                       # leave >= 0.20 for kickoffs
        scale = 0.80 / (ad + fr)
        ad *= scale
        fr *= scale
    c["air_dribble_w"] = round(ad, 4)
    c["flip_reset_w"] = round(fr, 4)
    c["kickoff_w"] = round(1.0 - ad - fr, 4)


def propose(best_cfg: Dict[str, Any], rng) -> Tuple[Dict[str, Any], str]:
    """Return (candidate_cfg, description) by stepping ONE random knob by +/- step."""
    cand = copy.deepcopy(best_cfg)
    path, lo, hi, step = TUNABLES[rng.randrange(len(TUNABLES))]
    cur = get_path(cand, path)
    direction = rng.choice((-1.0, 1.0))
    new = _clamp(cur + direction * step, lo, hi)
    if new == cur:                            # at a bound; flip direction
        new = _clamp(cur - direction * step, lo, hi)
    set_path(cand, path, round(new, 4))
    normalize_curriculum(cand)
    return cand, f"{path}: {cur} -> {get_path(cand, path)}"


# ---- persistence ----------------------------------------------------------
def load_config(path: str | None) -> Dict[str, Any]:
    """Load a config JSON, falling back to DEFAULT_CONFIG if missing/unset.

    Used by v4_env.build_env inside trainer subprocesses (path comes from the
    V4_LOOP_CONFIG env var), so it must never raise on a missing file.
    """
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            loaded = json.load(f)
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        _deep_update(cfg, loaded)             # tolerate partial/old configs
        normalize_curriculum(cfg)
        return cfg
    return copy.deepcopy(DEFAULT_CONFIG)


def save_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> None:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
