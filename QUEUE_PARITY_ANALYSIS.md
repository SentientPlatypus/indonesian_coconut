# Ripple Action Queue: Python vs Rust Side-by-Side Analysis

**Date**: 2026-11-08  
**Purpose**: Detailed comparison to understand regression (22W→10W at e7f38ea)

---

## 1. Queue Structure & Indexing

### Python (act.py lines 90-117)
```python
class DelayedAction:
    def __init__(self, parser, action_queue: np.ndarray, end_index: int = -1):
        """
        :param action_queue: Shared with obs builder, shape (6, controlled_players, 8)
                             NaNs mark unknown; first selection fills entire column
        """
        self.action_queue = action_queue  # Shape: (6, N, 8)
        self.end_index = end_index  # = -1, so queue[5]
```

**Queue indices**:
- `queue[0]` = ???
- `queue[5]` (or `queue[-1]`) = executed action (oldest)
- `end_index = -1` → `queue[-1]` = `queue[5]`

### Rust (ripple.rs current)
```rust
struct RippleState {
    queue: [[f32; 8]; QUEUE_LEN],  // QUEUE_LEN = 6
}
```

**At 38b2576a (baseline, 22W)**:
- After `rotate_left(1)`: moves queue[0] to queue[5], shifts everything left
- Executes `queue[0]` after rotate
- New action stored at `queue[5]`

**At e7f38ea (my "fix", 10W)**:
- Saves `to_execute = queue[5]` before rotate
- `rotate_right(1)`: moves queue[5] to queue[0], shifts everything right
- New action stored at `queue[0]`
- Executes saved `to_execute`

---

## 2. Execution Logic: What Index Gets Applied

### Python (act.py lines 109-117)
```python
def parse_actions(self, actions, state):
    new_actions = self.parser.parse_actions(actions, state)
    p = slice(0, new_actions.shape[0])
    
    # 1. Fill entire queue if NaN detected
    invalid = np.isnan(self.action_queue[:, p])
    if invalid.any():
        invalid_players = np.where(invalid.any(axis=(0, 2)))[0]
        for i in invalid_players:
            self.action_queue[:, i] = new_actions[i]  # Fill all 6 slots
    
    # 2. Save action to execute (BEFORE shift)
    next_actions = self.action_queue[self.end_index][p].copy()  # queue[5]
    
    # 3. Shift queue right: elements 0-4 move to positions 1-5
    self.action_queue[1:, p] = self.action_queue[:-1, p]
    
    # 4. Put new action at front
    self.action_queue[0, p] = new_actions
    
    # 5. Return saved action
    return next_actions  # Returns what WAS at queue[5]
```

**Key insight**: Returns `queue[5]` BEFORE the shift, so the action executed is the one that was at position 5.

### Rust Baseline (38b2576a, 22W)
```rust
let was_first_fill = st.queue[0][0].is_nan();
if was_first_fill {
    // Fill entire queue
    for slot in &mut st.queue {
        *slot = new_action;
    }
} else {
    // Shift queue left: [q0,q1,q2,q3,q4,q5] → [q1,q2,q3,q4,q5,?]
    st.queue.rotate_left(1);
    // Put new at end: [q1,q2,q3,q4,q5,new]
    st.queue[QUEUE_LEN - 1] = new_action;
}

// Execute from front AFTER rotate
let mut applied_action_controls = st.queue[0];  // = old q1
```

**Key insight**: After rotate_left, `queue[0]` contains what was at `queue[1]` before the rotate. So the executed action is the one that was at position 1.

### Rust "Fix" (e7f38ea, 10W)
```rust
if st.queue[0][0].is_nan() {
    for slot in &mut st.queue {
        *slot = new_action;
    }
}

// Save action to execute BEFORE rotate
let to_execute = st.queue[QUEUE_LEN - 1];  // queue[5]

// Shift right: [q0,q1,q2,q3,q4,q5] → [q5,q0,q1,q2,q3,q4]
st.queue.rotate_right(1);

// Put new at front: [new,q0,q1,q2,q3,q4]
st.queue[0] = new_action;

// Execute saved value
let mut applied_action_controls = to_execute;  // = old q5
```

**Key insight**: Executes what was at `queue[5]` before rotate, matching Python's approach.

---

## 3. Queue Order in Observations Tensor

### Python (obs.py line 238)
```python
# In pre_step(), after building player arrays:
actions[i] = self.action_queue[:, i, :]  # Copy entire queue as-is
```

Then at line 206:
```python
balls[:] = self.ball_predictions[::self.ball_step_size_ticks]
```

And at line 283-284:
```python
self.current_obs = tuple(np.ascontiguousarray(x, dtype=np.float32) for x in
                         (main_players, other_players, balls, actions, boosts))
```

**After Python's queue update**:
```
queue[0] = newest action (just decided)
queue[1] = 2nd newest
...
queue[5] = oldest action (about to be executed)
```

So the observations tensor has **newest-to-oldest** order.

### Rust Baseline (38b2576a)

In `build_obs_internal` (line ~490):
```rust
let mut actions = [[0.0f32; 8]; 6];
for (i, src) in st.queue.iter().enumerate() {
    for (j, &val) in src.iter().enumerate() {
        actions[i][j] = if val.is_nan() { 0.0 } else { val };
    }
}
```

**After baseline's queue update** (rotate_left + append at [5]):
```
queue[0] = 2nd oldest (was at [1])
queue[1] = 3rd oldest (was at [2])
...
queue[5] = newest (just added)
```

So observations tensor has **old-to-new** order - **BACKWARDS from Python!**

### Rust "Fix" (e7f38ea)

Same observation building code, but after the "fixed" update (rotate_right + prepend at [0]):
```
queue[0] = newest (just added)
queue[1] = 2nd newest (was at [0])
...
queue[5] = 5th newest (was at [4])
```

So observations tensor has **newest-to-old** order - **matches Python!**

---

## 4. Delay Timing Analysis

### Python
- `TICK_SKIP = 4` (from obs.py line 60)
- Ripple decides every 4 ticks
- Queue length = 6
- Action at queue[5] gets executed immediately
- Action at queue[0] gets executed after: 5 more decisions × 4 ticks = 20 ticks

**Delay**: An action decided now is put at queue[0] and executed 20 ticks later (after moving through 5 queue positions).

### Rust Baseline (rotate_left)
- Our framework calls with `tick_skip = 8` 
- But Ripple has `TICK_SKIP = 4` internally
- **CRITICAL QUESTION**: How many ticks between Ripple infer() calls in our arena?

If called every 8 ticks:
- Action put at queue[5] (newest position after rotate)
- Executed from queue[0] 
- After rotate_left(1), queue[0] contains what was at queue[1]
- **Delay**: 5 decisions × 8 ticks = 40 ticks (if we call every 8 ticks)

If called every 4 ticks:
- **Delay**: 5 decisions × 4 ticks = 20 ticks (matches Python)

### Rust "Fix" (rotate_right)
- Action put at queue[0] (newest position)
- Executed from queue[5] (saved before rotate)
- After rotate_right, the action at queue[0] will reach queue[5] after 5 more decisions
- **Delay**: 5 decisions × tick_skip

---

## 5. Reset/NaN Fill Logic

### Python (act.py lines 110-114)
```python
invalid = np.isnan(self.action_queue[:, p])
if invalid.any():
    invalid_players = np.where(invalid.any(axis=(0, 2)))[0]
    for i in invalid_players:
        self.action_queue[:, i] = new_actions[i]  # Fill ALL 6 slots
```

**Then STILL does the shift and add**: So after first fill:
1. Queue = `[new, new, new, new, new, new]`
2. Save `next = queue[5]` = `new`
3. Shift right
4. Put `new` at queue[0]
5. Return `new`

Result: First action is executed immediately, queue remains all same value.

### Python Kickoff Init (obs.py line ~130-133)
```python
# In reset():
if (state.ball.position[0] == 0 and state.ball.position[1] == 0
    and all(np.all(abs(p.car_data.linear_velocity) < 10) for p in initial_state.players)):
    self.action_queue[:] = 0  # Kickoffs start with all no-op
else:
    self.action_queue[:] = np.nan
```

So at **kickoff**, Python fills with **zeros**, not NaN!

### Rust Baseline (38b2576a, after commit 71286b3)
```rust
if needs_reset {
    *st = RippleState::new();
    // Leave queue as NaN so infer() fills it with the first selected action
}

// In infer():
if st.queue[0][0].is_nan() {
    // Fill entire queue on first decision
    for slot in &mut st.queue {
        *slot = new_action;
    }
}
```

Then continues with rotate_left + execute queue[0].

**Difference from Python**: Rust leaves as NaN and fills on first decision. Python fills with 0 at kickoff in reset().

### Rust "Fix" (e7f38ea)
Same NaN initialization, but the fill + execute logic should match Python better since it executes from queue[5].

---

## 6. Decision Cadence

### Python Environment
- `TICK_SKIP = 4`
- Ripple's `decide()` is called every 4 ticks by the RLGym environment

### Our Rust Framework
- Framework's `tick_skip = 8` (our trainer setting)
- Ripple's internal `TICK_SKIP = 4`
- **Question**: Does `action_delay()` affect how often we call Ripple?

In ripple.rs:
```rust
fn action_delay(&self) -> u8 {
    0  // No delay
}
```

So we call Ripple's `build()` and `infer()` every 8 ticks (framework tick_skip).

**MISMATCH**: Python calls Ripple every 4 ticks, we call every 8 ticks!

This means:
- Python: 6-slot queue @ 4 ticks = 24 ticks total pipeline
- Rust: 6-slot queue @ 8 ticks = 48 ticks total pipeline

The delay is **doubled** in our environment!

---

## 7. Summary: Why the Regression?

### Baseline (38b2576a, 22W)
- ❌ Observations: queue order backwards (old→new instead of new→old)
- ❌ Execution: Wrong delay (executes position 1 instead of position 5)
- ✓ But: Works somewhat because the backwards queue might accidentally compensate for other issues?

### "Fix" (e7f38ea, 10W)
- ✓ Observations: queue order correct (new→old matching Python)
- ✓ Execution: Executes from position 5 (matching Python)
- ❌ But: Made things worse!

### Hypotheses for Regression

1. **Decision cadence mismatch**: We call Ripple every 8 ticks, Python calls every 4. This doubles all delays. The "fix" might make this mismatch more apparent.

2. **The two wrongs made a right**: The baseline's backwards queue + wrong execution position might have accidentally created the right delay for our 8-tick cadence.

3. **NaN vs Zero initialization**: Python fills with 0 at kickoff, we fill with NaN then the first decision. This might affect kickoff behavior.

4. **Incomplete fix**: The observations are now correct, but we might need to adjust the delay to account for our 8-tick cadence vs Python's 4-tick cadence.

---

## 8. Recommended Testing

The EC2 2×2 ablation will show:
- Does ball step 30 vs 120 affect results?
- Does queue order old vs new affect results?

If "new" queue order is worse, possible fixes:
1. Keep observations order correct but adjust execution delay
2. Implement `action_delay()` to call Ripple every 4 ticks instead of 8
3. Adjust queue length from 6 to 3 to compensate for doubled tick cadence

---

**Conclusion**: The "fix" made observations match Python but exposed the decision cadence mismatch (4 ticks vs 8 ticks). The baseline's bugs might have accidentally compensated for this mismatch.
