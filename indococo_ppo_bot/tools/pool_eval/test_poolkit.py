"""Fast checks of the pool harness (no bot weights needed).

    python tools/pool_eval/test_poolkit.py   (or pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from keep_best import should_promote  # noqa: E402
from pool_eval import pool_score, summarize  # noqa: E402
from poolkit.agent import RandomAgent, ZeroAgent  # noqa: E402
from poolkit.match import Arena  # noqa: E402
from poolkit.v1compat import spectator_ids, to_v1  # noqa: E402
from poolkit.view import AgentView  # noqa: E402


def test_summary_and_score():
    r = summarize("x", [1, 1, -1, 0])
    assert (r["wins"], r["losses"], r["timeouts"]) == (2, 1, 1)
    assert abs(r["win_rate_decisive"] - 2 / 3) < 1e-3
    assert pool_score([{"win_rate_decisive": 0.2}, {"win_rate_decisive": 0.6}]) == 0.4


def test_keep_best_rule():
    assert should_promote(0.3, None, None)
    best = {"pool_score": 0.5}
    assert not should_promote(0.4, best, None)
    assert not should_promote(0.6, best, 0.48)
    assert should_promote(0.6, best, 0.55)


def test_zero_vs_zero_times_out_and_v1_view_is_consistent():
    arena = Arena()
    # Two idle cars never touch the ball: 30 s no-touch timeout -> 0.
    assert arena.play(ZeroAgent(), ZeroAgent(), seed=1) == 0
    state = arena.engine.create_base_state()
    arena.mutator.apply(state, {})
    state = arena.engine.set_state(state, {})
    cid = next(iter(state.cars))
    view = AgentView(state, cid, {c: -10**9 for c in state.cars}, 0, True)
    gs, me = to_v1(view)
    assert sorted(spectator_ids(view).values()) == [1, 5]
    assert me.team_num == state.cars[cid].team_num
    assert np.allclose(me.car_data.position, state.cars[cid].physics.position)
    assert np.allclose(me.inverted_car_data.position[:2], -state.cars[cid].physics.position[:2])
    assert gs.boost_pads.shape == (34,) and gs.boost_pads.all()


def test_random_agent_returns_table_rows():
    a = RandomAgent()
    assert a.decide(None).shape == (8,)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
