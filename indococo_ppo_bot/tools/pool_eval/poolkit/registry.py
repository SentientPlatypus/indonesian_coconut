"""Builds agents from pool entries: `{name, kind, path, ...options}`."""
from __future__ import annotations

import os

from .agent import Agent, RandomAgent, ZeroAgent


def make_agent(kind: str, path: str = "", **options) -> Agent:
    """`options` are extra keys of the pool entry (e.g. `clamp_air_time` for indococo)."""
    path = os.path.expanduser(path)
    if kind == "indococo":
        from .candidate import IndococoPolicy

        return IndococoPolicy(path, **options)
    if kind in ("nexto", "necto", "element", "immortal"):
        from . import bots

        return {"nexto": bots.Nexto, "necto": bots.Necto, "element": bots.Element,
                "immortal": bots.Immortal}[kind](path)
    if kind == "zero":
        return ZeroAgent()
    if kind == "random":
        return RandomAgent()
    raise ValueError(f"unknown agent kind {kind!r}")
