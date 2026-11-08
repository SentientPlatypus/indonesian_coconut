# Ripple Parity Status - In Progress

## Commits
- `0ae82371` Fix rotation matrix transpose and boost scale
- `cf3e68b1` Add Python/Rust parity harness  
- `ca682f00` Fix timer edge detection
- `3a0e68ef` Fix pos/vel swap, queue init, ball predictions

## Parity Progress (Blue side, 175 steps)

### Fixed Bugs
1. ✓ Rotation matrix transpose (row vs column major)
2. ✓ Boost scale (0-100 → 0-1 normalization)  
3. ✓ Position/velocity in main tensor
4. ✓ Timer edge detection (prev_prev_action)
5. ✓ Ball prediction updates

### Current Parity Results
- **Action match: 6.3%** (was 0%, now 11/175)
- Max diffs:
  - main: 1.0 (on_ground - replay-only, uses wheel contacts)
  - others: 1.85 (coordinate transform issues)
  - ball: 1.46 (coordinate transform issues)
  - actions: 2.0 (action queue/history)
  - boosts: 2.0 (boost pad positions)

### Idle Opponent Test (200 games)
- **Win rate: 75%** (21W-7L-172D)
- **No improvement** from rotation/boost fixes
- 86% draws - Ripple times out, can't score
- Python baseline: 100% wins, 7.2s avg

## Remaining Issues

### 1. Coordinate Transformations
**Symptoms**: others/ball/boosts tensors have 1.4-2.0 diffs
**Cause**: Relative position calculations after rotation/inversion not matching Python
**Impact**: Wrong spatial awareness → wrong actions

### 2. Action Queue/History  
**Symptoms**: actions tensor has 2.0 diff
**Cause**: Queue initialization, NaN handling, or delay logic mismatch
**Impact**: Action history affects decision → wrong actions

### 3. On-Ground (Replay Only)
**Symptoms**: main[17] differs by 1.0 in replay
**Cause**: flags() checks wheel contacts, not preserved when setting states from JSON
**Impact**: Replay only - live training uses natural evolution with correct contacts

## Next Steps to Reach 90% Idle Win Rate

The 75% win rate despite fixes suggests the main issue is **not** rotation/boost bugs.
More likely causes:

1. **Coordinate transform bugs** - others/ball positions wrong by ~2 units
2. **Action history bugs** - past actions fed to model are wrong  
3. **Normalization bugs** - some values on wrong scale
4. **Side/team inversion bugs** - flipping logic errors

All of these cause wrong ONNX inputs → wrong policy outputs → can't score.

## Plan Forward

1. Fix coordinate transformation to match Python exactly
2. Fix action queue/history handling
3. Debug normalization (NORM vector application)
4. Re-test idle opponent after each fix
5. Once ≥90% idle wins, run 50-game h2h

Python gets 100-0-0 in 7.2s avg. Rust at 75% with 86% draws means
something fundamental is still broken in the observations.
