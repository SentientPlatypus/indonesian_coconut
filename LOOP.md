# V4 self-improving loop (driven by Claude Code `/loop`)

The **brain** is Claude running in `/loop`; the **hands** are two scripts:
`freestyler_v4.py` (train a phase) and `eval_match.py` (measure strength vs the
Element bot). Each loop iteration trains a while, evaluates, then keeps or
reverts a single reward/curriculum tweak. State lives in `data/loop_state/`.

- **Yardstick:** `Rlgym-v2-to-rlbot-v5/src/element_killer.pt` (fallback `47-3.pt`).
- **Search:** eval-gated hill-climb — one clamped knob per phase (`loop_config.TUNABLES`).
- **Safety:** training always resumes from the tracked *best* checkpoint, so a
  bad tweak is discarded; an all-time-best guardrail prevents slow drift.

---

## 0. SMOKE TEST — run this ONCE before launching the loop (do not skip)

None of this was testable on WSL; the smoke test validates the two unknowns
(opponent obs compatibility, flip-reset granting) on the real EC2 box. ~10 min.

```bash
cd <repo>
mkdir -p data/loop_state
# (a) tiny training phase: confirm it loads V3, runs curriculum, saves to V4
V4_SAVE_DIR=data/checkpoints/V4_smoke timeout 300 python freestyler_v4.py \
  > data/loop_state/smoke_train.log 2>&1
ls data/checkpoints/V4_smoke/*/PPO_POLICY.pt        # a checkpoint must exist
# (b) eval that checkpoint vs Element (4 games) — VALIDATES OBS COMPATIBILITY
python eval_match.py \
  --candidate "$(ls -d data/checkpoints/V4_smoke/*/ | sort -n | tail -1)PPO_POLICY.pt" \
  --opponent Rlgym-v2-to-rlbot-v5/src/element_killer.pt --games 4
```

Pass criteria:
1. `(a)` saved a checkpoint and the log shows no traceback.
2. `(b)` printed a JSON result (no `AssertionError: opponent input ... != obs`).
   - **If the assertion fires**, `element_killer.pt` uses a different observation
     than DefaultObs → it is NOT usable as the yardstick. **STOP and tell the
     user.** Fall back to the frozen V3 policy as opponent:
     `--opponent "$(ls -d data/checkpoints/V3/17.9B/*/ | sort -n | tail -1)PPO_POLICY.pt"`
3. In `smoke_train.log`, the **FlipReset** reward channel is **not flat at 0**.
   If it is, RocketSim isn't granting resets from the curriculum spawn — lower
   `curriculum.flip_reset_w` and lean on air dribbles (note it to the user).
4. **Airdribble** and **WallPopSetup** channels are alive too (the curriculum
   now includes a `wall_pop` entry state, and the Airdribble dense rate was
   fixed from an unintended 8x dilution — both should be clearly nonzero).

Delete `data/checkpoints/V4_smoke` afterwards.

---

## 1. Launch the loop (on EC2, inside tmux)

```bash
tmux new -s indococo            # persists across SSH drops
claude                          # then, in Claude:
```
In Claude, start the self-paced loop:
```
/loop Run ONE iteration of the procedure in @LOOP.md, then schedule the next.
```
(No interval → self-paced; you pace phases with `ScheduleWakeup`.)

---

## 2. State file — `data/loop_state/state.json`

Create on iteration 0 if missing. Schema:
```json
{
  "iteration": 0,
  "phase_running": false,
  "phase_started_at": null,
  "phase_target_seconds": 7200,
  "pre_phase_latest_ts": null,
  "best_ckpt": null,
  "best_score": null,
  "best_style": null,
  "best_ever_ckpt": null,
  "best_ever_score": null,
  "below_best_streak": 0,
  "opponent": "Rlgym-v2-to-rlbot-v5/src/element_killer.pt",
  "eval_games": 60,
  "last_change": null
}
```
- `best_ckpt: null` ⇒ first phase resumes from V3 (handled by `freestyler_v4.py`).
- `current_config.json` = config for the running phase; `best_config.json` = the
  config that produced `best_ckpt`. Initialize both from defaults:
  `python -c "import loop_config as L,json; L.save_config(L.DEFAULT_CONFIG,'data/loop_state/best_config.json'); L.save_config(L.DEFAULT_CONFIG,'data/loop_state/current_config.json')"`

---

## 3. Per-iteration procedure (each `/loop` fire)

Read `state.json` first; recover whatever phase you're in.

> **CAPABILITY MODE** (`state.mode == "airdribble_capability"`): the win-rate
> hill-climb below is SUSPENDED. We found (300-game evals + `tools/eval_airdribble_spawn.py`)
> that win-rate vs Element is orthogonal to the real goal ("air dribbles when
> advantageous"), and the bot can't carry the ball (~3% completion from an ideal
> spawn). This mode runs ONE long dedicated air-dribble training run
> (`data/loop_state/airdribble_config.json`, curriculum ~80% air-dribble,
> resumed from the frozen 0.233 policy) and gates on **spawn-eval capability**,
> not win-rate. Each `/loop` wake:
> 1. If training is running, leave it running. Spawn-eval the NEWEST complete
>    `data/checkpoints/V4/*` checkpoint (read-only; training keeps going):
>    `python tools/eval_airdribble_spawn.py --candidate <newest>/PPO_POLICY.pt --episodes 120 --out data/loop_state/spawn_<ts>.json`
> 2. Append `completed_frac`/`engaged_frac` to `data/loop_state/capability.csv`.
>    If `completed_frac` beats `best_completed_frac`, snapshot that checkpoint to
>    `data/checkpoints/V4_best/cap_<ts>` and update `best_capability_ckpt`/`best_completed_frac`.
> 3. If `completed_frac` climbs → keep training, `ScheduleWakeup` ~3600s.
>    If it's flat for ~4 checks (plateau) or training died → stop, summarize to
>    the user (the mechanic may need a reward-shape fix, i.e. reward the CARRY
>    not just the touch — see the deferred "carry reward" investigation).
> Skip §A/§B below while in this mode.

> **INTEGRATION MODE** (`state.mode == "integration"`): the air-dribble carry
> mechanic is trained (~0.13 spawn completion, 4x baseline). Now teach the bot to
> use it *when advantageous* without losing strength. ONE long run
> (`data/loop_state/integration_config.json`: scoring-dominant rewards + LIGHT
> freestyle, MIXED curriculum ~55% kickoff / 30% air-dribble), resumed from the
> best air-dribble policy (`state.best_capability_ckpt`). Gate on TWO metrics each
> wake, on the newest complete `data/checkpoints/V4/*` checkpoint (training keeps
> running). **SNAPSHOT that checkpoint to `data/checkpoints/V4_best/intg_<ts>`
> BEFORE the evals** — the dual-eval takes ~20min and the checkpoint rotates out
> of the save dir meanwhile; eval the snapshot, not the live dir.
> 1. Win-rate: `python eval_match.py --candidate <newest>/PPO_POLICY.pt --opponent <state.opponent> --games 300 --out ...` — must stay >= `state.integration_winrate_floor` (0.18).
> 2. Capability: `python tools/eval_airdribble_spawn.py --candidate <newest>/PPO_POLICY.pt --episodes 200 --out ...` — completed_frac should stay >= `state.integration_capability_floor` (0.08).
> Append both to `data/loop_state/integration.csv`. Snapshot to
> `data/checkpoints/V4_best/intg_<ts>` when BOTH floors hold and win-rate improves.
> If win-rate recovers toward ~0.23 while capability stays >= floor → success,
> export+push. If freestyle rewards drag win-rate below floor → cut them further
> and note it. Skip §A/§B while in this mode.

### A. If no phase is running (`phase_running == false`)
1. **Pick the config for this phase:**
   - iteration 0: use `best_config.json` as-is (baseline, to set `best_score`).
   - iteration ≥ 1: perturb ONE knob from `best_config.json`. Either reason about
     which knob to try (respecting `loop_config.TUNABLES` bounds) and write it, or:
     `python -c "import loop_config as L,random,json; c=L.load_config('data/loop_state/best_config.json'); cand,desc=L.propose(c,random.Random()); L.save_config(cand,'data/loop_state/current_config.json'); print(desc)"`
     Record `desc` in `state.last_change`.
2. **Record** `pre_phase_latest_ts` = newest numbered dir in `data/checkpoints/V4`
   (or null). Set `V4_RESUME_DIR` = `best_ckpt` (omit if null).
3. **Launch training in the background** (Bash `run_in_background: true`).
   MUST wrap in `script` to allocate a PTY: `rlgym_ppo`'s `Learner` builds a
   `KBHit()` that calls `termios.tcgetattr(stdin)`. A backgrounded Bash job has
   stdin=`/dev/null` (no TTY) → `termios.error: (25) Inappropriate ioctl` and the
   learn loop dies after 0 new steps. `script -q -c '<cmd>' <log>` gives the child
   a pseudo-terminal so KBHit succeeds. (The foreground smoke test never hits this
   because it has a real terminal.)
   ```bash
   script -q -c 'env \
     V4_LOOP_CONFIG=data/loop_state/current_config.json \
     V4_RESUME_DIR=<best_ckpt or omit this line> \
     V4_SAVE_DIR=data/checkpoints/V4 \
     V4_WANDB_GROUP=<session tag, e.g. loop_2026-07-03> \
     V4_WANDB_RUN=iter_<iter> \
     python freestyler_v4.py' data/loop_state/train_<iter>.log
   ```
   (`n_proc` defaults to 28 for the 32-vCPU g6.8xlarge; wandb is ON by
   default — the group tag stays fixed for the whole loop session so every
   iteration stacks in one wandb view. Set `V4_WANDB=0` to disable.)
4. Set `phase_running=true`, `phase_started_at=$(date +%s)`, persist state.
5. `ScheduleWakeup` ~3600s (re-check; phase target is longer than one wake).

### B. If a phase is running
1. `elapsed = $(date +%s) - phase_started_at`.
2. If `elapsed < phase_target_seconds`: confirm the training process is still
   alive (if it died, inspect `train_<iter>.log`, alert if it's a real error),
   then `ScheduleWakeup` ~3600s and stop.
3. If `elapsed >= phase_target_seconds`:
   a. **Stop training** (kill the background process / pid).
   b. **Candidate** = newest numbered dir in `data/checkpoints/V4` with all 5
      files present (`PPO_POLICY.pt`, both optimizers, value net, BOOK_KEEPING).
      It must be newer than `pre_phase_latest_ts`; if not, the phase produced no
      new checkpoint → extend the phase or alert.
   c. **Evaluate:**
      ```bash
      python eval_match.py --candidate <candidate>/PPO_POLICY.pt \
        --opponent <state.opponent> --games <state.eval_games> \
        --out data/loop_state/eval_<iter>.json
      ```
   d. Read `score` (= P(candidate scores | a goal)), `margin`, and the style
      metrics (`style` = air dribbles/game, `cand_flip_resets`) from the JSON.
   e. **Decide** (see §4), update state + `best_config.json`, append a row to
      `data/loop_state/history.csv`
      (`iteration,last_change,score,margin,decided,style,flip_resets,decision`).
   e2. **Protect the new best from rotation (CRITICAL).** The training save dir
      keeps only the last 5 checkpoints, so `best_ckpt` gets DELETED once the
      next phase trains ~5M steps past it — which silently breaks resume (the
      loop then feeds a dead path to the loader and the phase dies with an empty
      log). On any PROMOTE (or style-tie PROMOTE), snapshot the FULL candidate
      dir to the rotation-safe `data/checkpoints/V4_best/` and point state at it:
      ```bash
      SNAP=data/checkpoints/V4_best/$(basename <candidate>)
      mkdir -p "$SNAP" && cp <candidate>/* "$SNAP"/
      # set state.best_ckpt (and best_ever_ckpt if best-ever) = "$SNAP"
      ```
      `_resolve_resume_dir` now hard-errors on a missing resume dir, so a
      regression here fails loud instead of producing a zero-byte train log.
   f. **On PROMOTE** (and at least every 5 iterations): refresh the pull-and-
      test folder and push it so the user can test locally —
      ```bash
      python tools/export_best.py
      git add checkpoints_to_test && git commit -m "Export V4 best (iter <N>)" && git push
      ```
   g. `iteration += 1`, `phase_running=false`, persist, then go to step A
      (or `ScheduleWakeup` ~60s to start the next phase promptly).

---

## 4. Decision policy (noise-aware; ~60 trials ⇒ margin σ ≈ 8 goals)

Let `s` = candidate score, `b` = `best_score`, `y` = candidate `style`
(air dribbles/game), `by` = `best_style`.

The objective is LEXICOGRAPHIC: strength first (never give up win-rate beyond
noise), then style. Win-rate alone would silently optimize air dribbles away —
that is exactly what the 2B-timestep V4 run did.

- iteration 0: set `best_score=s`, `best_style=y`, `best_ckpt=candidate`, also
  `best_ever_*=` these.
- **Clear regression** (`s < b - 0.05`): **ROLLBACK** — leave `best_ckpt`/
  `best_config` unchanged (next phase resumes from best, discarding the
  candidate). `below_best_streak += 1`.
- **Clear improvement** (`s >= b + 0.03`): **PROMOTE** — `best_ckpt=candidate`,
  `best_score=s`, `best_style=y`, copy `current_config.json` → `best_config.json`.
  `below_best_streak=0`.
- **Neutral score** (within noise) — style breaks the tie:
  - `y >= by + 0.15` (clearly more air dribbles): treat as **PROMOTE** — the
    knob bought style without costing strength. Update `best_*` incl. config.
  - `y <= by - 0.15` (style clearly lost): **keep the training, drop the knob**
    — `best_ckpt=candidate`, `best_score=s` but leave `best_config.json` AND
    `best_style` unchanged.
  - otherwise: **keep the training, drop the knob** (as above), and set
    `best_style=max(by, y)`.
- Update `best_ever_*` whenever `s` beats `best_ever_score`.

**Style stagnation:** if `style == 0` for 4 consecutive iterations, stop
sampling knobs uniformly — pick from the mechanic set only
(`reward_weights.airdribble`, `airdribble_seq`, `wall_pop`,
`reward_weights.aerial_boost` UP, `reward_weights.boost_change` DOWN,
`curriculum.wall_pop_w`, `curriculum.air_dribble_w`, `ppo_ent_coef` upward)
and note it in `last_change`.

**Known failure mode (the pre-loop 2B run):** the bot farmed wall pops —
kick up wall, pop, hop — without boosting to the ball, because the pop paid
per-touch with no completion requirement while three boost-economy rewards
punished the aerial commit. The pop follow-bonus is now touch-gated, but if
eval shows `cand_air_touch_steps` near 0 while training WallPop channels are
hot, that pattern is back: push `aerial_boost` up / `boost_change` down first.

**Guardrails (halt/escalate — don't silently continue):**
- If `below_best_streak >= 2` **or** `best_score < best_ever_score - 0.08`:
  hard-rollback `best_ckpt=best_ever_ckpt`, restore `best_config` from the
  best-ever config, reset streak, and **post a summary to the user**.
- If a borderline result (`|s - b| < 0.03`) would flip a decision, optionally
  re-eval once at `--games 120` before deciding.
- Never edit a knob outside `loop_config.TUNABLES` bounds. `loop_config` keeps
  `kickoff_w >= 0.20` automatically.
- If eval errors (opponent incompatible, no checkpoint, etc.): **pause the loop
  and tell the user** — never promote on a missing/failed eval.

---

## 5. Monitoring / stopping
- `data/loop_state/history.csv` — one row per iteration (plot to see the climb).
- `checkpoints_to_test/STATUS.md` — committed summary; on your local machine
  `git pull` and test `checkpoints_to_test/PPO_POLICY_V4_BEST.pt` in RLBot.
- `train_<iter>.log` — per-phase training output (reward channels).
- **wandb (ON by default):** run `wandb login` ONCE on the box before the
  first phase (a headless run with no login errors out). Each phase is its own
  run `iter_<N>` under group `V4_WANDB_GROUP`; compare reward/PPO curves across
  iterations there. Set `V4_WANDB=0` on a launch to disable.
- **Stop:** end the `/loop` (don't schedule the next wakeup), or in tmux Ctrl-C.
  The best policy is always `state.best_ever_ckpt`.
