# URGENT: Isolate Regression

**Current Status**: Changes at e17397f made idle test WORSE  
- Before (38b2576a): 22W-0L-78D  
- After (e17397f): 10W-0L-90D (blue 0-0-50, orange 10-0-40)  

One or both of the following changes caused regression:
1. **Queue order fix** (e7f38ea): rotate_right instead of rotate_left  
2. **Ball prediction fix** (5545142): BALL_STEP_TICKS 30→120  

## Test Procedure (on working EC2 instance)

### Step 1: Verify Baseline
```bash
git checkout 38b2576a
cargo build --release --features ripple
# Run idle test - should get ~22W
```

### Step 2: Test Ball Prediction Change ONLY
```bash
git checkout 38b2576a
# Manually change ONLY line 38 in src/ripple.rs:
#   const BALL_STEP_TICKS: usize = 120;  // was 30
cargo build --release --features ripple
# Run idle test - if this gets ~10W, ball fix is wrong; if ~22W, it's OK
```

### Step 3: Test Queue Order Change ONLY  
```bash
git checkout 38b2576a
# Manually apply ONLY the queue changes from e7f38ea (lines 880-920 in src/ripple.rs)
cargo build --release --features ripple
# Run idle test - if this gets ~10W, queue fix is wrong; if ~22W, it's OK
```

### Step 4: Test Both Together
```bash
git checkout e17397f
# This should match the 10W result
```

## Expected Outcomes

- If ball fix alone → 10W: **Ball fix is wrong**, revert to 30  
- If queue fix alone → 10W: **Queue fix is wrong/incomplete**, revert or fix
- If both separately OK but together bad: **Interaction bug**  

## Hypothesis

The queue order change might be correct for OBSERVATIONS but wrong for EXECUTION.

Check in infer() at line 903:
```rust
let to_execute = st.queue[QUEUE_LEN - 1];  // queue[5]
```

Is this being executed BEFORE or AFTER the rotate? The Python code executes queue[5], THEN shifts. Make sure Rust does the same.

Also check: after the initial NaN fill, does the queue get shifted properly? Line 894-909 in e7f38ea.

## Quick Fix Candidates

If queue order is the problem, try:
1. Keep rotate_right but execute queue[0] AFTER rotate (not queue[5] BEFORE)  
2. Or: execute queue[5] before rotate, don't shift, just overwrite old values
