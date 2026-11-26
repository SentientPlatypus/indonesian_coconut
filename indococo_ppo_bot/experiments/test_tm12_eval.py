#!/usr/bin/env python3
"""Unittests for experiments/tm12_eval.py (stdlib only)."""
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("tm12_eval", HERE / "tm12_eval.py")
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
            cand = Path(d) / "cand"
            fb = Path(d) / "fb"
            for ck in (cand, fb):
                ck.mkdir()
                (ck / "actor.bpk").write_text("x")
                (ck / "critic.bpk").write_text("x")
                (ck / "stats.toml").write_text("x")
            # Report panel: pass (mean 0.51, min 0.50, nexto in band)
            report = Path(d) / "report.json"
            report.write_text(json.dumps({
                "h2h_2v2": 0.52, "h2h_3v3": 0.50,
                "nexto_2v2": 0.649, "nexto_3v3": 0.528,
            }))
            r = E.start(str(cand), str(fb), report=str(report))
            self.assertEqual(r["start"], str(cand), r)
            # mean 0.49 fails
            report.write_text(json.dumps({
                "h2h_2v2": 0.50, "h2h_3v3": 0.48,
                "nexto_2v2": 0.649, "nexto_3v3": 0.528,
            }))
            r = E.start(str(cand), str(fb), report=str(report))
            self.assertEqual(r["start"], str(fb), r)
            # mode 0.47 fails even if mean >= 0.50
            report.write_text(json.dumps({
                "h2h_2v2": 0.55, "h2h_3v3": 0.47,
                "nexto_2v2": 0.649, "nexto_3v3": 0.528,
            }))
            r = E.start(str(cand), str(fb), report=str(report))
            self.assertEqual(r["start"], str(fb), r)
            # nexto above the reference is allowed (one-sided gate)
            report.write_text(json.dumps({
                "h2h_2v2": 0.52, "h2h_3v3": 0.50,
                "nexto_2v2": 0.70, "nexto_3v3": 0.528,
            }))
            r = E.start(str(cand), str(fb), report=str(report))
            self.assertEqual(r["start"], str(cand), r)
            # nexto more than the band below the reference fails
            report.write_text(json.dumps({
                "h2h_2v2": 0.52, "h2h_3v3": 0.50,
                "nexto_2v2": 0.60, "nexto_3v3": 0.528,
            }))
            r = E.start(str(cand), str(fb), report=str(report))
            self.assertEqual(r["start"], str(fb), r)
            # missing nexto fails
            report.write_text(json.dumps({"h2h_2v2": 0.52, "h2h_3v3": 0.50}))
            r = E.start(str(cand), str(fb), report=str(report))
            self.assertEqual(r["start"], str(fb), r)
            # incomplete cand
            bad = Path(d) / "bad"
            bad.mkdir()
            (bad / "critic.bpk").write_text("x")
            (bad / "stats.toml").write_text("x")
            r = E.start(str(bad), str(fb), report=str(report))
            self.assertEqual(r["start"], str(fb))
            # mid path: one mid is not enough
            mid = Path(d) / "mid.jsonl"
            mid.write_text(json.dumps({"ckpt": str(cand), "h2h_2v2": 0.55, "h2h_3v3": 0.50,
                                       "nexto_2v2": 0.649, "nexto_3v3": 0.528}) + "\n")
            r = E.start(str(cand), str(fb), mid_path=str(mid))
            self.assertEqual(r["start"], str(fb))
            # two mids, mins (0.52, 0.48) mean 0.50, nexto in band
            mid.write_text("\n".join([
                json.dumps({"ckpt": "a", "h2h_2v2": 0.55, "h2h_3v3": 0.50, "nexto_2v2": 0.65, "nexto_3v3": 0.53}),
                json.dumps({"ckpt": "b", "h2h_2v2": 0.52, "h2h_3v3": 0.48, "nexto_2v2": 0.64, "nexto_3v3": 0.52}),
            ]) + "\n")
            r = E.start(str(cand), str(fb), mid_path=str(mid))
            self.assertEqual(r["start"], str(cand), r)
            # two mids, 3v3 min 0.47 fails
            mid.write_text("\n".join([
                json.dumps({"ckpt": "a", "h2h_2v2": 0.55, "h2h_3v3": 0.50, "nexto_2v2": 0.65, "nexto_3v3": 0.53}),
                json.dumps({"ckpt": "b", "h2h_2v2": 0.52, "h2h_3v3": 0.47, "nexto_2v2": 0.64, "nexto_3v3": 0.52}),
            ]) + "\n")
            r = E.start(str(cand), str(fb), mid_path=str(mid))
            self.assertEqual(r["start"], str(fb), r)
            # direct panel args
            r = E.start(str(cand), str(fb), h2h=(0.52, 0.50), nexto=(0.649, 0.528))
            self.assertEqual(r["start"], str(cand), r)

    def test_mid_nexto_structure_keys(self):
        with tempfile.TemporaryDirectory() as d:
            W = Path(d)
            # complete required evals (wr needs games and wins+losses)
            def wr(name, games=200):
                (W / f"{name}.json").write_text(json.dumps({
                    "games": games, "wins": 100, "losses": 80, "win_rate_decisive": 0.555
                }))
            for n in ("h2h_2v2", "h2h_3v3", "gal_2v2", "gal_3v3", "gal_1v1", "mega_2v2",
                      "h2h_1v1_st25", "nexto_2v2", "nexto_3v3", "necto_2v2", "necto_3v3"):
                wr(n, 200 if "h2h" in n or "nexto" in n or n.startswith("gal_") else 104)
            audit = [
                {"label": "nexto_2v2",
                 "kickoff": {"same_pad_two_nontakers": 0.1,
                             "outcome_1.5s(ball y +-800)": {"win": 0.15, "loss": 0.41, "neutral": 0.44},
                             "outcome_3s(ball y +-1500)": {"win": 0.2, "loss": 0.3, "neutral": 0.5},
                             "goal_8s_against": 0.01},
                 "live": {"dc_two_within_1000_closing500": 0.025, "pair800": 0.09},
                 "concede": {"pair800_in_last_3s": 0.32, "double_commit_in_last_3s": 0.11,
                             "open_play_concedes_per_game": 0.4,
                             "goalside_cars_1s_before": {"0": 8, "1": 7, "2": 1}}},
                {"label": "nexto_3v3",
                 "kickoff": {"same_pad_two_nontakers": 0.2,
                             "outcome_1.5s(ball y +-800)": {"win": 0.15, "loss": 0.41, "neutral": 0.44},
                             "outcome_3s(ball y +-1500)": {"win": 0.2, "loss": 0.3, "neutral": 0.5},
                             "goal_8s_against": 0.02},
                 "live": {"dc_two_within_1000_closing500": 0.041, "pair800": 0.186},
                 "concede": {"pair800_in_last_3s": 0.61, "double_commit_in_last_3s": 0.2,
                             "open_play_concedes_per_game": 0.5,
                             "goalside_cars_1s_before": {0: 4, 1: 6}}},
            ]
            (W / "kickoff_audit.jsonl").write_text("\n".join(json.dumps(x) for x in audit) + "\n")
            necto = [{"label": "necto_2v2",
                      "kickoff": {"outcome_1.5s(ball y +-800)": {"win": 0.3, "loss": 0.2, "neutral": 0.5},
                                  "outcome_3s(ball y +-1500)": {"win": 0.4, "loss": 0.2, "neutral": 0.4}}},
                     {"label": "necto_3v3",
                      "kickoff": {"outcome_1.5s(ball y +-800)": {"win": 0.25, "loss": 0.25, "neutral": 0.5},
                                  "outcome_3s(ball y +-1500)": {"win": 0.3, "loss": 0.3, "neutral": 0.4}}}]
            (W / "necto_kickoff_audit.jsonl").write_text("\n".join(json.dumps(x) for x in necto) + "\n")
            for n in (2, 3):
                (W / f"h2h_{n}v{n}.json").write_text(json.dumps({
                    "games": 200, "wins": 100, "losses": 80, "win_rate_decisive": 0.555,
                    "team": {"a": {k: 0.1 for k in (
                        "passes_per_min", "double_commit_ratio", "pair_lt800_ratio",
                        "two_near_ball1500_ratio", "double_commit_wide_ratio",
                        "deep_last_man_ratio", "no_challenger_ratio", "role_swaps_per_min",
                        "touch_share_first", "first_mean_stint_s", "touches_per_car_min",
                        "deepest_car_depth_uu", "demos_per_min", "bumps_per_min")}}
                }))
            m = E.mid(str(W), 100, "/ckpt", {
                "h2h_2v2": 200, "h2h_3v3": 200, "gal_2v2": 200, "gal_3v3": 200,
                "gal_1v1": 200, "mega_2v2": 104, "nexto_2v2": 200, "nexto_3v3": 200,
                "necto_2v2": 104, "necto_3v3": 104, "h2h_1v1_st25": 200,
            })
            self.assertTrue(m["complete"], m.get("missing"))
            self.assertEqual(m["nx_dc_2"], 0.025)
            self.assertEqual(m["nx_pair800_3"], 0.186)
            self.assertEqual(m["nx_conc_pair800_2"], 0.32)
            self.assertEqual(m["nx_conc_dc_3"], 0.2)
            self.assertAlmostEqual(m["nx_zero_goalside_2"], 8 / 16)
            self.assertAlmostEqual(m["nx_zero_goalside_3"], 4 / 10)
            self.assertEqual(m["nx_open_conc_2"], 0.4)
            self.assertEqual(m["necto_ko_15s_2v2"]["win"], 0.3)
            # nx keys are reported, not required
            (W / "kickoff_audit.jsonl").unlink()
            m2 = E.mid(str(W), 100, "/ckpt", {
                "h2h_2v2": 200, "h2h_3v3": 200, "gal_2v2": 200, "gal_3v3": 200,
                "gal_1v1": 200, "mega_2v2": 104, "nexto_2v2": 200, "nexto_3v3": 200,
                "necto_2v2": 104, "necto_3v3": 104, "h2h_1v1_st25": 200,
            })
            self.assertTrue(m2["complete"], m2.get("missing"))
            self.assertIsNone(m2["nx_dc_2"])


    def test_entropy_alarm_one_sided_floor(self):
        with tempfile.TemporaryDirectory() as d:
            mid = Path(d) / "mid.jsonl"
            # two mids, entropy missing unless metrics provided
            mid.write_text("\n".join([
                json.dumps({"h2h_2v2": 0.55, "h2h_3v3": 0.55, "passes_min_2v2": 1.0, "passes_min_3v3": 1.5,
                            "deep_2v2": 0.1, "deep_3v3": 0.1, "nexto_2v2": 0.6, "nexto_3v3": 0.5,
                            "near2_2v2": 0.2, "near2_3v3": 0.2}),
                json.dumps({"h2h_2v2": 0.55, "h2h_3v3": 0.55, "passes_min_2v2": 1.0, "passes_min_3v3": 1.5,
                            "deep_2v2": 0.1, "deep_3v3": 0.1, "nexto_2v2": 0.6, "nexto_3v3": 0.5,
                            "near2_2v2": 0.2, "near2_3v3": 0.2, "entropy": 3.0}),
            ]) + "\n")
            # without ENTROPY_ALARM, no entropy rule
            self.assertNotIn("entropy", E.alarm(str(mid), {}))
            # mid lines lack entropy on the first row -> intersection won't include entropy
            # provide entropy on both rows
            mid.write_text("\n".join([
                json.dumps({"entropy": 3.0, "h2h_2v2": 0.55, "h2h_3v3": 0.55, "passes_min_2v2": 1.0,
                            "passes_min_3v3": 1.5, "deep_2v2": 0.1, "deep_3v3": 0.1, "nexto_2v2": 0.6,
                            "nexto_3v3": 0.5, "near2_2v2": 0.2, "near2_3v3": 0.2}),
                json.dumps({"entropy": 3.1, "h2h_2v2": 0.55, "h2h_3v3": 0.55, "passes_min_2v2": 1.0,
                            "passes_min_3v3": 1.5, "deep_2v2": 0.1, "deep_3v3": 0.1, "nexto_2v2": 0.6,
                            "nexto_3v3": 0.5, "near2_2v2": 0.2, "near2_3v3": 0.2}),
            ]) + "\n")
            hits = E.alarm(str(mid), {"ENTROPY_ALARM": "3.2"})
            self.assertIn("entropy", hits)
            mid.write_text("\n".join([
                json.dumps({"entropy": 3.5, "h2h_2v2": 0.55, "h2h_3v3": 0.55, "passes_min_2v2": 1.0,
                            "passes_min_3v3": 1.5, "deep_2v2": 0.1, "deep_3v3": 0.1, "nexto_2v2": 0.6,
                            "nexto_3v3": 0.5, "near2_2v2": 0.2, "near2_3v3": 0.2}),
                json.dumps({"entropy": 3.6, "h2h_2v2": 0.55, "h2h_3v3": 0.55, "passes_min_2v2": 1.0,
                            "passes_min_3v3": 1.5, "deep_2v2": 0.1, "deep_3v3": 0.1, "nexto_2v2": 0.6,
                            "nexto_3v3": 0.5, "near2_2v2": 0.2, "near2_3v3": 0.2}),
            ]) + "\n")
            hits = E.alarm(str(mid), {"ENTROPY_ALARM": "3.2"})
            self.assertNotIn("entropy", hits)

if __name__ == "__main__":
    unittest.main()
