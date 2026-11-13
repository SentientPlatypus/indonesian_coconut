# tools/viz: metrics and figures for the ST chain

`make_all.py` is a post-hoc plotter. It only reads files the trainer, launcher and diag scripts already write, and it never makes up a number. If a panel has no data, it is skipped and the skip is logged.

| Source | Written by | Used for |
|---|---|---|
| `exp_<RUN>/logs/metrics.jsonl` (167 keys per PPO iteration) | trainer (`INDOCOCO_METRICS`) | training curves, reward breakdown, behaviour, setter mix, cross-run compare |
| `exp_<RUN>/logs/mid.jsonl` | `st_launch.sh` after each 100M chunk | mid-eval trajectories (200 g vs BEST/FR4/Galactus/Megabyte, 95% Wilson CI) |
| `st<run>_final/{h2h_*,pool400,bp5,galactus_final,megabyte_final}.jsonl`, `PROMOTE.json` | finals + `st_promote.py` | leaderboard heatmap, gate marks, gen-panel radar, bp5 bars |
| `diag_gal/st<run>_summary.txt` | `diag_gal/analyze.py` | post-demo concede share, aerial goals against, demo exchange, win rate by goal time |
| `diag_gal/st<run>_gal200.jsonl` | `diag_gal/diag.py` | where Galactus demos us (car position from the ~6 s traj window kept before each goal) |

```bash
# On EC2 (coco-venv has numpy 1.26.4 + matplotlib; ~10 s, single CPU core, nice 19)
cd /opt/dlami/nvme/ic_ppo/indococo_ppo_bot
/home/ubuntu/coco-venv/bin/python tools/viz/make_all.py \
    --runs ST19r,ST20r,ST21,ST22,ST23,ST24,ST25,ST26,ST27,ST28,ST29,ST30,ST31,ST32,ST33 \
    --current ST32,ST33 --best ST25 --out docs/figures --summary-json docs/figures/summary.json
```

Flags:
- `--runs-dir`: defaults to `~/ippo_runs`.
- `--runs`: defaults to every `st*_final` from ST19 on.
- `--current`: per-run plots for each run listed. Defaults to the last run.
- `--best`: defaults to the newest promoted `*_final` by mtime, the same rule the launcher uses.
- `--diag-runs`: runs to include in the Galactus diag panels.
- `--heat-runs`: runs for the demo heatmap. Defaults to ST25,ST31,ST32.

Figures:
- `train_curves_<RUN>.png`: throughput, entropy, policy and value loss, approx KL, clip fraction, GAE return, update magnitude and episode length. Shows raw values plus a rolling mean, with BEST's run dashed for comparison. Explained variance is not logged by the trainer.
- `reward_breakdown_<RUN>.png`: the top-10 `RewardAbs/*` terms, stacked, plus each term's % share.
- `behavior_<RUN>.png`: events per minute (goals, aerial goals, demos, demoed, flip resets, air-dribble bumps), player state, and the state-setter mix.
- `mid_eval_<RUN>.png`: per-chunk evals with 95% CI. The 52% gate, BEST's finals and the Galactus gate (BEST − 4) are drawn as lines.
- `mid_eval_overlay.png`: the last nine runs aligned by M steps into the run. BEST and the current run are highlighted.
- `leaderboard_heatmap.png`: runs × finals panel. Colour is Δ vs BEST, a red box means the cell fails the current gate, ★ means promoted, and "(declined)" marks a user override.
- `bp5_radar.png`: radar of the 12-opponent gen panel, plus botpack5 per-bot bars with 95% CI.
- `runs_compare.png`: entropy, KL, demoed/min, goals/min, |face_ball| and steps/s across runs.
- `galactus_diag.png`: how we concede, demo exchange, win rate by deciding-goal time (hollow points have n<10), and win rate in all games vs games where we got demoed.
- `demo_heatmap.png`: where Galactus demos us, oriented with our goal at the bottom. White dots mark demos followed by a goal against within 3.5 s.

The launcher hook in `experiments/st_launch.sh` runs this after each run's finals. It runs detached, at nice 19, with a 15-minute timeout, and writes to `~/ippo_runs/viz/<RUN>/`. A failure there can't block the chain.

## 3D positioning (`pos_dump.py` + `make_positions.py`)

`diag_gal` only keeps the last ~6 s before each goal, so `pos_dump.py` replays games using the same protocol (poolkit kickoff, first goal wins, both sides per seed). It records the ball and both cars over the whole game: position, boost, on-ground and demoed. A frame is written each time the sim steps forward, which is every 8 ticks (15 Hz) because that's the agents' action repeat. It runs on CPU only and needs no new binary.

```bash
cd /opt/dlami/nvme/ic_ppo/indococo_ppo_bot
for o in galactus fr4; do
  CUDA_VISIBLE_DEVICES= nice -n 19 /home/ubuntu/coco-venv/bin/python tools/viz/pos_dump.py \
    --cand ~/ippo_runs/st25_final/final_policy.safetensors --opp $o --seeds 100 --procs 8 \
    --out ~/ippo_runs/viz/positions/st25_vs_$o.jsonl        # 200 games, ~1 min each
done
nice -n 19 /home/ubuntu/coco-venv/bin/python tools/viz/make_positions.py --label ST25 \
  --in galactus=~/ippo_runs/viz/positions/st25_vs_galactus.jsonl \
  --in fr4=~/ippo_runs/viz/positions/st25_vs_fr4.jsonl --out docs/figures/positions
```

Everything is rotated so our goal is at −y. Demoed frames are dropped, and so is the first 3 s of each game (`--skip-kickoff`), because the kickoff paths would otherwise dominate the density. Outputs:
- `positions_3d.html`: interactive plotly volume over a field wireframe with walls, goals and the 2044 ceiling. The dropdown switches between us, the opponent and the ball, vs Galactus or vs FR4. plotly.js loads from the CDN and the data is inline (about 420 KB).
- `positions_3d.png`: matplotlib 3D voxels from three angles, for each matchup.
- `positions_topdown.png`: log-density for us, the opponent, and us while the opponent attacks. "Attacks" means the ball is in our half, moving at our goal (vy < −300), and the opponent is closer to it.
- `positions_height.png`: side-view y×z density plus height bands.
- `positions_stats.json`: % of time goal-side, airborne, z>300, in our own half, and closer to the ball. During attacks: % goal-side, % in our box, and mean distance to our goal.
