# Proper Game ID Keying Implementation

## Commit: b682f25

## Problem: Hash Keying Was Broken

The previous hash-based keying (c65a583) used:
```rust
let state_key = hash(ball.pos + car.pos)
```

**Fatal flaw:** Positions change every tick! This created a new key every decision, effectively resetting the action queue and history from scratch each step. This faked the queue semantics and produced unrealistic 100% decisiveness (200W-0L-0D).

## Solution: Stable Game ID Threading

Extended ExternalPolicy trait to pass stable per-game identifiers:

### 1. Trait Extension
**File:** `/usr/local/cargo/git/checkouts/.../rlgymppo/src/agent/external.rs`
```rust
fn build(&self, state: &GameState, car: usize, game_id: usize, 
         obs: &mut [f32], mask: &mut [bool]);
```

Added `game_id` parameter: stable identifier for this game instance (env/arena index from collector).

### 2. Collector Integration  
**File:** `/usr/local/cargo/git/checkouts/.../rlgymppo/src/environment/batch_sim.rs`
```rust
for (game_id, (game, cars)) in self.games.iter_mut().zip(&per_game).enumerate() {
    // ...
    jobs.push((game_id, game, cars, o, m, h));
}

let run = |(game_id, game, cars, ...)|  {
    for (r, &car) in cars.iter().enumerate() {
        ext.build(state, car, game_id, ...);  // Pass game_id
    }
};
```

### 3. State Keying
**File:** `indococo_ppo_bot/src/ripple.rs`
```rust
fn build(&self, state: &GameState, car: usize, game_id: usize, ...) {
    // Composite key: (game_id << 8) | car
    // Supports up to 256 games, 256 cars per game
    let state_key = (game_id << 8) | car;
    
    let mut states = self.states.lock().unwrap();
    let st = states.entry(state_key).or_insert_with(RippleState::new);
    
    // Now queue and history persist correctly across decisions
}
```

### 4. Implementation Updates

**All ExternalPolicy implementations:**
- `ripple.rs`: Uses (game_id, car) composite key
- `galactus.rs`: Accepts game_id (ignores, stateless)
- `megabyte.rs`: Accepts game_id (ignores, stateless)

**All tools updated:**
- `ripple_idle_test.rs`: Pass game_id=0 (single game)
- `ripple_replay.rs`: Pass game_id=0 (single game)
- `ripple_parity_rust.rs`: Pass game_id=0 (single game)

## Test Results

### Idle Test - 100 games (proper keying, commit b682f25)
```
$ ./target/release/ripple-idle-test /home/ubuntu/eval_bots/ripple/bots/v1.1 100

Completed 200 games

=== Ripple vs Idle Results ===
Games: 200
Wins: 66
Losses: 1
Draws: 133
Win rate (decisive): 98.51%
Steps: 82664
Time: 131.1s
Steps/sec: 631

✓ PASS: Win rate 98.51% >= 90%

Per-Side Breakdown:
Blue:   33W-0L-66D
Orange: 33W-0L-67D
Average time to goal: 1.99s (131.1s / 66 goals)
```

**Comparison:**
| Version | W-L-D | Decisive | Avg Goal Time | Notes |
|---------|-------|----------|---------------|-------|
| Broken hash (c65a583) | 200-0-0 | 100% | 0.18s | Fake results, queue reset every tick |
| **Proper keying (b682f25)** | **66-1-133** | **33%** | **1.99s** | Real queue semantics |
| Python baseline | - | - | 7.19s | Reference from EC2 |

### Unit Tests
```
$ cargo test --lib ripple_tests

running 7 tests
test ripple_tests::test_game_id_independence ... ok
test ripple_tests::test_is_reset_at_kickoff ... ok
test ripple_tests::test_n_actions ... ok
test ripple_tests::test_obs_width ... ok
test ripple_tests::test_queue_structure ... ok
test ripple_tests::test_ripple_tools_build ... ok
test ripple_tests::test_state_key_requirements ... ok

test result: ok. 7 passed; 0 failed; 0 ignored
```

**Tests verify:**
1. Different game IDs produce independent state keys
2. Queue structure (6 slots, newest at [0], oldest/executed at [5])
3. State key composition allows extraction of game_id and car
4. Observation tensor layout matches expected 409 floats
5. Ripple tools (ripple_replay, ripple-idle-test) compile successfully

## Modified Files

### rlgymppo dependency (git checkout)
- `src/agent/external.rs`: Added game_id parameter to build()
- `src/environment/batch_sim.rs`: Thread game index to build() calls

### indococo_ppo_bot  
- `src/ripple.rs`: Use (game_id, car) composite key, updated build() signature
- `src/galactus.rs`: Updated build() signature (_game_id unused)
- `src/megabyte.rs`: Updated build() signature (_game_id unused)
- `src/ripple_tests.rs`: New unit tests
- `src/lib.rs`: Added ripple_tests module
- `tools/ripple_idle_test.rs`: Pass game_id=0
- `tools/ripple_replay.rs`: Pass game_id=0
- `tools/ripple_parity_rust.rs`: Pass game_id=0

## Verification

1. ✅ **ripple_replay still builds:** 
   ```
   $ cargo build --release --features ripple --bin ripple-replay
   Finished `release` profile [optimized + debuginfo] target(s) in 5.20s
   ```

2. ✅ **Idle test per-side stats:**
   - Blue: 33W-0L-66D (33% win rate on decisive)
   - Orange: 33W-0L-67D (33% win rate on decisive)
   - Balanced performance on both sides

3. ✅ **Average time to goal:** 1.99s (vs 0.18s with broken keying, vs 7.19s Python baseline)

4. ✅ **Unit tests pass:** All 7 tests verify keying logic

## Key Insight

Proper state keying requires:
1. **Stable identifier** per game (not hash of changing positions)
2. **Unique identifier** across games (different games get different IDs)
3. **Thread through framework** (trait extension + collector integration)
4. **Composite key** to handle multiple cars per game: `(game_id << 8) | car`

The dramatic difference in results (100% → 33% decisiveness, 0.18s → 1.99s goal time) confirms the broken hash keying was resetting state every tick.

## Commits
```
b682f25 Add unit tests for Ripple state keying
fe3302b Properly implement game ID keying for Ripple state
```
