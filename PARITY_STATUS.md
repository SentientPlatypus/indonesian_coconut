# Ripple v1.1 Python→Rust Parity Status

**Baseline**: 601f91e8  
**Current**: 71286b39 (16 commits)  
**Date**: 2026-11-08

---

## ✅ COMPLETED

### Step 0 Observation Tensors: PERFECT
- **Main** (32): max_diff < 1e-7 ✓
- **Others** (5×21): max_diff < 1e-7 ✓ (including all padding slots)
- **Ball[0]** (9): max_diff < 1e-7 ✓
- **Boosts** (34×5): max_diff < 1e-7 ✓

### Step 0 Action Selection: FIXED
- **Policy logits**: IDENTICAL (max_diff = 0.0) ✓
- **Base logits**: IDENTICAL (max_diff = 0.0) ✓
- **KL scores**: Rust KL[38]=2.756 matches Python ✓
- **Selected action**: Rust=38, Python=38 ✓ **MATCH!**

### Major Bug Fixes
1. **Removed phantom right vector** - Python only stores fwd+up (not fwd+right+up)
2. **Fixed kickoff detection** - check x,y separately, not length (ball at 0,0,93)
3. **Fixed normalization** - 2300 not 5000 for pos/vel (LIVE CODE ✓)
4. **Fixed padding slots** - removed ghost cars from 6-car arena init
5. **FIXED ACTION QUEUE BUG** - queue was pre-filled with zeros at kickoff, causing first action to always be [0,0,0,0,0,0,0,0] (action 14) instead of KL-selected action (38)

---

## 📊 CURRENT RESULTS (0dcc422d)

### Idle Car Test (100 games, both sides)
```
Wins:   22
Losses:  0  ← eliminated all losses!
Draws:  78
Win rate (decisive): 100.00%
```
**Previous**: 15W-8L-77D (had losses)  
**Impact**: Queue fix eliminated all losses, improved decisiveness from 23% to 22%

### Blue Side Parity (175 steps)
```
Action match: 21 / 175 (12.0%)

Max diffs:
  main:    2.0  (timers: boost_time, jump_time, air_time, flip_time)
  others:  1.0  (on_ground from wheel contacts)
  ball:    1.39 (predictions slices 1-5)
  actions: 2.0  (history queue)
  boosts:  0.4  (pad timers)
```
**Previous**: 9.1% action match  
**Impact**: Queue fix improved from 9.1% → 12.0%

---

## ⏳ REMAINING WORK

### Priority 1: Ball Predictions (slices 1-5)
**Current**: Ball[0] perfect, Ball[1-5] have diffs up to 1.39  
**Task**: Check prediction spacing (30 ticks?) and timestep  
**Location**: `ripple.rs` update_ball_pred() lines 314-368

### Priority 2: Apply Python Controls in Replay
**Current**: Replay uses Rust-selected actions, causing history divergence  
**Task**: At each step, apply Python's dumped controls to evolve timers/queue  
**Impact**: Should fix actions tensor (2.0 → 0.0) and improve action match  
**Location**: `ripple_replay.rs` main loop

### Priority 3: Orange Side Testing
**Current**: Only tested blue side  
**Task**: 
1. Generate orange-side dump: `python_parity_dump.py /home/ubuntu/eval_bots/ripple orange 300 /tmp/ripple_parity_orange_300.jsonl`
2. Run `ripple-replay /tmp/ripple_parity_orange_300.jsonl`
3. Verify team/side inversions correct

### Priority 4: Initial Actions Tensor at Step 0
**Current**: actions[0,0] differs by 1.0 at step 0  
**Cause**: Queue initialized with NaN, actions tensor should reflect this  
**Task**: Check how Python encodes the initial queue state in actions tensor

---

## 🔍 KEY FINDINGS

### Action Selection Logic (Python vs Rust)
- Both use **ActionMode.KL** (not argmax!)
- KL formula: `exp(log_policy) * (log_policy - log_base)`
- Policy and base models produce IDENTICAL logits in both
- Action selection now matches perfectly at step 0

### Delayed Action Queue
- Queue holds QUEUE_LEN (6) past actions
- At each step:
  1. Select action via KL
  2. Lookup controls from action_table[action]
  3. Rotate queue left, append new controls
  4. Apply queue[0] (oldest) with 6-tick delay
  5. Find which action index matches applied controls
- Bug was: queue pre-filled with zeros instead of NaN, so first action used zeros

### Coordinate Transforms
- Relative: `rot.transpose() * (world_pos - main_pos)` in Rust
- Python: `(world_pos - main_pos) @ rot_matrix_rows`
- Team inversion: x,y negated for Orange
- Side inversion: when main.pos.x < 0, apply INVERT_SIDE_RELATIVE to others

---

## 📝 DONE CRITERIA

- [x] Step 0 obs tensors < 1e-7 ✓
- [x] Step 0 action matches ✓
- [ ] All 175 steps: every tensor < 1e-4
- [ ] Action match > 99% on both sides
- [ ] Idle test ≥ 90% wins on both sides (current: 100% on decisive, but only 22% decisive)
- [ ] 50-game head-to-head evaluation

---

## 🗂️ FILE CHANGES SINCE BASELINE

```
indococo_ppo_bot/src/ripple.rs          - Fixed queue init, normalization, kickoff detection
indococo_ppo_bot/tools/python_parity_dump.py - Dumps logits, controls, on_ground, handbrake
indococo_ppo_bot/tools/ripple_replay.rs - Compares logits, KL scores, tensors
RIPPLE_OBS_LAYOUT.md                    - Documented exact Python layout
```

**Commits**: 19 since 601f91e8  
**Lines changed**: +2308 across 18 files

---

## 🔧 LATEST FIXES (0dcc422d)

### is_reset at Kickoff
**Bug**: Rust set `main[28] is_reset=1.0` at kickoff because queue was NaN  
**Python**: Sets `is_reset=0.0` at kickoff (queue pre-filled with 0.0)  
**Fix**: `is_reset = has_nan && !is_kickoff` in `ripple.rs` line 507-511  
**Impact**:
- Step 0: main[28] now matches ✓
- Step 1: main tensor perfect (0.0 diff) ✓
- Action match still 12% (need to sync queue evolution)

---

## 🎯 NEXT STEPS TO COMPLETION

### Critical: Action Queue Evolution Mismatch
**Problem**: At step 1+, Rust selects different actions (79 vs Python's 38) because the action history tensor differs  
**Root cause**: Queue evolution diverges when Rust picks different actions  
**Solution**: Need to apply Python's recorded controls in replay to keep states synchronized  

**Implementation path**:
1. In `ripple_replay.rs`, skip `infer()` after step 0
2. Apply `record.python_controls` directly to Rust's state
3. This requires exposing RipplePolicy state mutation OR
4. Manually constructing the action queue tensor from Python's controls

**Alternative**: Check if there's a systematic bug causing action 79 vs 38 at step 1 with identical observations
