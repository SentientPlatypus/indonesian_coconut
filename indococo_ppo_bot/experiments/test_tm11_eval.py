#!/usr/bin/env python3
"""Unittests for experiments/tm11_eval.py (stdlib only)."""
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("tm11_eval", HERE / "tm11_eval.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


def wr_json(games, wins, losses, rate=None):
    if rate is None:
        rate = wins / (wins + losses) if wins + losses else 0.5
    return {"games": games, "wins": wins, "losses": losses, "win_rate_decisive": rate}


def write_eval(d, name, payload):
    p = Path(d) / f"{name}.json"
    p.write_text(json.dumps(payload) + "\n")
    return p


def h2h_payload(rate, games=200, wins=None, losses=None, team_a=None):
    if wins is None:
        wins = int(round(rate * games))
        losses = games - wins
    d = wr_json(games, wins, losses, rate)
    a = {
        "passes_per_min": 1.2,
        "double_commit_ratio": 0.03,
        "pair_lt800_ratio": 0.1,
        "two_near_ball1500_ratio": 0.15,
        "double_commit_wide_ratio": 0.02,
        "deep_last_man_ratio": 0.5,
        "no_challenger_ratio": 0.03,
        "role_swaps_per_min": 4.0,
        "touch_share_first": 0.5,
        "first_mean_stint_s": 3.0,
        "touches_per_car_min": 12.0,
        "deepest_car_depth_uu": 2500.0,
        "demos_per_min": 0.4,
        "bumps_per_min": 1.1,
    }
    if team_a:
        a.update(team_a)
    d["team"] = {"a": a}
    d["roles"] = {"margin": 300, "hold": 8, "obs_tiebreak": True, "touch_mode": "hysteresis"}
    return d


def complete_mid_dir(d, rates=None, expect_games=200):
    rates = rates or {}
    def r(name, default):
        return rates.get(name, default)
    write_eval(d, "h2h_2v2", h2h_payload(r("h2h_2v2", 0.60), games=expect_games))
    write_eval(d, "h2h_3v3", h2h_payload(r("h2h_3v3", 0.55), games=expect_games))
    for name, default in (
        ("gal_2v2", 0.20), ("gal_3v3", 0.18), ("gal_1v1", 0.50),
        ("mega_2v2", 0.10), ("h2h_1v1_st25", 0.55),
        ("nexto_2v2", 0.58), ("nexto_3v3", 0.48),
        ("necto_2v2", 0.95), ("necto_3v3", 0.94),
    ):
        g = 200 if "h2h" in name or "nexto" in name or name.startswith("gal_") else 104
        if name == "mega_2v2":
            g = 104
        if name == "h2h_1v1_st25":
            g = expect_games
        write_eval(d, name, wr_json(g, int(r(name, default) * g), g - int(r(name, default) * g), r(name, default)))
    return d


def ckpt(td, name="ck"):
    p = Path(td) / name
    p.mkdir(exist_ok=True)
    for f in ("actor.bpk", "critic.bpk", "stats.toml"):
        (p / f).write_text("x")
    return str(p)


class TestTm11Eval(unittest.TestCase):
    def test_mid_complete_score_ignores_necto(self):
        with tempfile.TemporaryDirectory() as d:
            complete_mid_dir(d)
            m = E.mid(d, "100", "/ck/a", {})
            self.assertTrue(m["complete"])
            self.assertEqual(m["missing"], [])
            h = (0.60 + 0.55) / 2
            e = (0.58 + 0.48 + 0.20 + 0.18) / 4
            self.assertEqual(m["score"], round(0.5 * h + 0.5 * e, 4))
            necto = m["necto_2v2"]
            self.assertIsNotNone(necto)
            # Changing necto files must not change the score.
            write_eval(d, "necto_2v2", wr_json(104, 0, 104, 0.0))
            m2 = E.mid(d, "100", "/ck/a", {})
            self.assertEqual(m2["score"], m["score"])

    def test_mid_eval_fail_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            complete_mid_dir(d)
            write_eval(d, "gal_3v3", {})
            m = E.mid(d, "100", "/ck/a", {})
            self.assertIsNone(m["gal_3v3"])
            self.assertEqual(m["missing"], ["gal_3v3"])
            self.assertIsNone(m["score"])
            self.assertFalse(m["complete"])

    def test_wr_short_zero_or_nan(self):
        self.assertIsNone(E.wr(wr_json(50, 30, 20, 0.6), expect=200))
        self.assertIsNone(E.wr({"games": 200, "wins": 0, "losses": 0, "win_rate_decisive": 0.5}, expect=200))
        self.assertIsNone(E.wr({"games": 200, "wins": 100, "losses": 100, "win_rate_decisive": float("nan")}, expect=200))
        self.assertAlmostEqual(E.wr(wr_json(200, 120, 80, 0.6), expect=200), 0.6)

    def test_pick_prefers_complete_and_ignores_incomplete_high(self):
        with tempfile.TemporaryDirectory() as d:
            mid = Path(d) / "mid.jsonl"
            mid.write_text("\n".join([
                json.dumps({"ckpt": "/ck/60", "h2h_2v2": 0.60, "h2h_3v3": 0.60,
                            "nexto_2v2": 0.5, "nexto_3v3": 0.5, "gal_2v2": 0.2, "gal_3v3": 0.2}),
                json.dumps({"ckpt": "/ck/high", "h2h_2v2": 0.99, "h2h_3v3": 0.99,
                            "nexto_2v2": 0.99, "nexto_3v3": None, "gal_2v2": 0.99, "gal_3v3": 0.99}),
                json.dumps({"ckpt": "/ck/58", "h2h_2v2": 0.58, "h2h_3v3": 0.58,
                            "nexto_2v2": 0.5, "nexto_3v3": 0.5, "gal_2v2": 0.2, "gal_3v3": 0.2}),
            ]) + "\n")
            self.assertEqual(E.pick(str(mid)), "/ck/60")
            mid.write_text(json.dumps({"ckpt": "/ck/x", "h2h_2v2": 0.9, "h2h_3v3": None,
                                       "nexto_2v2": 0.5, "nexto_3v3": 0.5, "gal_2v2": 0.2, "gal_3v3": 0.2}) + "\n")
            self.assertEqual(E.pick(str(mid)), "")

    def test_guard_low_missing_ok(self):
        with tempfile.TemporaryDirectory() as d:
            mid = Path(d) / "mid.jsonl"
            mid.write_text(json.dumps({"h2h_2v2": 0.50, "h2h_3v3": 0.39}) + "\n")
            self.assertEqual(E.guard(str(mid)), "LOW")
            mid.write_text(json.dumps({"h2h_2v2": 0.50, "h2h_3v3": None}) + "\n")
            self.assertEqual(E.guard(str(mid)), "MISSING")
            mid.write_text(json.dumps({"h2h_2v2": 0.41, "h2h_3v3": 0.45}) + "\n")
            self.assertEqual(E.guard(str(mid)), "OK")
            self.assertEqual(E.guard(str(Path(d) / "empty.jsonl")), "MISSING")

    def test_alarm_intersection(self):
        with tempfile.TemporaryDirectory() as d:
            mid = Path(d) / "mid.jsonl"
            mid.write_text("\n".join([
                json.dumps({"h2h_2v2": 0.5, "h2h_3v3": 0.5, "nexto_2v2": 0.5}),
                json.dumps({"h2h_2v2": 0.5, "h2h_3v3": 0.5, "nexto_2v2": 0.5}),
            ]) + "\n")
            got = E.alarm(str(mid), {})
            self.assertIn("missing:passes_min_2v2", got)
            mid.write_text("\n".join([
                json.dumps({"h2h_2v2": 0.5, "h2h_3v3": 0.5, "nexto_2v2": 0.3, "passes_min_2v2": 1.0, "passes_min_3v3": 2.0,
                            "deep_2v2": 0.1, "deep_3v3": 0.1, "nexto_3v3": 0.5, "near2_2v2": 0.2, "near2_3v3": 0.2}),
                json.dumps({"h2h_2v2": 0.5, "h2h_3v3": 0.5, "nexto_2v2": 0.3, "passes_min_2v2": 1.0, "passes_min_3v3": 2.0,
                            "deep_2v2": 0.1, "deep_3v3": 0.1, "nexto_3v3": 0.5, "near2_2v2": 0.2, "near2_3v3": 0.2}),
            ]) + "\n")
            got = E.alarm(str(mid), {})
            self.assertIn("nexto2", got)
            mid.write_text("\n".join([
                json.dumps({"h2h_2v2": 0.5, "h2h_3v3": 0.5, "nexto_2v2": 0.3, "passes_min_2v2": 1.0, "passes_min_3v3": 2.0,
                            "deep_2v2": 0.1, "deep_3v3": 0.1, "nexto_3v3": 0.5, "near2_2v2": 0.2, "near2_3v3": 0.2}),
                json.dumps({"h2h_2v2": 0.5, "h2h_3v3": 0.5, "nexto_2v2": 0.5, "passes_min_2v2": 1.0, "passes_min_3v3": 2.0,
                            "deep_2v2": 0.1, "deep_3v3": 0.1, "nexto_3v3": 0.5, "near2_2v2": 0.2, "near2_3v3": 0.2}),
            ]) + "\n")
            got = E.alarm(str(mid), {})
            self.assertNotIn("nexto2", got)

    def test_gate_promote_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            # h2h uses expect 450
            for n, r in (("h2h_2v2", 0.55), ("h2h_3v3", 0.53)):
                write_eval(d, n, wr_json(504, int(r * 504), 504 - int(r * 504), r))
            for n, r in (
                ("gal_2v2", 0.12), ("gal_3v3", 0.11),
                ("gal_2v2_best", 0.10), ("gal_3v3_best", 0.09),
                ("nexto_2v2", 0.60), ("nexto_3v3", 0.50),
                ("nexto_2v2_best", 0.58), ("nexto_3v3_best", 0.48),
                ("necto_2v2", 0.95), ("necto_3v3", 0.94),
                ("necto_2v2_best", 0.95), ("necto_3v3_best", 0.94),
                ("gal_1v1", 0.5), ("mega_2v2", 0.1), ("h2h_1v1_st25", 0.55),
            ):
                g = 504 if n.startswith("h2h") else 256
                write_eval(d, n, wr_json(g, int(r * g), g - int(r * g), r))
            g = E.gate(d)
            self.assertTrue(g["promote_team"], g)
            os.remove(os.path.join(d, "nexto_3v3_best.json"))
            g2 = E.gate(d)
            self.assertFalse(g2["promote_team"])
            self.assertIn("nexto_3v3_best", g2["missing"])

    def test_start_promote_and_mid_rules(self):
        with tempfile.TemporaryDirectory() as d:
            cand = ckpt(d, "cand")
            fb = ckpt(d, "fb")
            prom = Path(d) / "TEAM_PROMOTE.json"
            prom.write_text(json.dumps({"h2h_2v2": 0.55, "h2h_3v3": 0.52}))
            r = E.start(cand, fb, promote=str(prom))
            self.assertEqual(r["start"], cand)
            prom.write_text(json.dumps({"h2h_2v2": 0.44, "h2h_3v3": 0.70}))
            r = E.start(cand, fb, promote=str(prom))
            self.assertEqual(r["start"], fb)
            prom.write_text(json.dumps({"h2h_2v2": 0.48, "h2h_3v3": 0.50}))
            r = E.start(cand, fb, promote=str(prom))
            self.assertEqual(r["start"], fb)
            prom.write_text(json.dumps({"h2h_2v2": 0.60}))
            r = E.start(cand, fb, promote=str(prom))
            self.assertEqual(r["start"], fb)
            # missing actor.bpk
            bad = Path(d) / "bad"
            bad.mkdir()
            (bad / "critic.bpk").write_text("x")
            (bad / "stats.toml").write_text("x")
            r = E.start(str(bad), fb, promote=str(prom))
            self.assertEqual(r["start"], fb)
            mid = Path(d) / "mid.jsonl"
            mid.write_text(json.dumps({"ckpt": cand, "h2h_2v2": 0.55, "h2h_3v3": 0.50}) + "\n")
            r = E.start(cand, fb, mid_path=str(mid))
            self.assertEqual(r["start"], fb)
            # Spec §5.3 wrote (0.55,0.50) and (0.52,0.47) -> mins (0.52,0.47) -> cand,
            # but start() (complete file) requires mean >= 0.50; (0.52+0.47)/2 = 0.495
            # fails. Use 0.48 so mins (0.52, 0.48) have mean 0.50 and min 0.45.
            mid.write_text("\n".join([
                json.dumps({"ckpt": "a", "h2h_2v2": 0.55, "h2h_3v3": 0.50}),
                json.dumps({"ckpt": "b", "h2h_2v2": 0.52, "h2h_3v3": 0.48}),
            ]) + "\n")
            r = E.start(cand, fb, mid_path=str(mid))
            self.assertEqual(r["start"], cand, r)
            mid.write_text("\n".join([
                json.dumps({"ckpt": "a", "h2h_2v2": 0.55, "h2h_3v3": 0.50}),
                json.dumps({"ckpt": "b", "h2h_2v2": 0.52, "h2h_3v3": 0.47}),
            ]) + "\n")
            r = E.start(cand, fb, mid_path=str(mid))
            self.assertEqual(r["start"], fb, r)


if __name__ == "__main__":
    unittest.main()
