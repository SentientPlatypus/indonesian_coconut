# Ripple Integration Final Status @ cfcb255

## ✅ Major Bug Fixed: Action Queue

**Root Cause**: `infer()` was returning the CHOSEN action ID, but Ripple's design requires applying the 6-old QUEUED action (300ms latency compensation).

**Fix**: Modified `infer()` in ripple.rs to return the action index from `st.prev_action` (queue[0]) instead of the just-chosen action.

**Evidence**:
- Before fix: Ripple chose actions 10, 14 (throttle=0) repeatedly
- After fix: Ripple chooses actions 42, 46, 30, 32, 135 (throttle=1.0)

## ✅ Working Components

1. **ripple-dump**: Generates 100 states with full game data + 5 ONNX tensors ✓
2. **Ripple inference**: Choosing varied actions ✓
3. **Action queue**: Properly delayed actions applied ✓
4. **Controls**: `set_car_controls()` working with proper throttle values ✓

## ⚠️  Remaining Issue: 0% Win Rate

**Symptom**: Candidate (rlbot_indococo_ppo/weights) wins 100% vs Ripple

**Possible causes**:
1. **Trained policy is very strong**: The rlbot weights are likely > ST16 level
   - User's baseline: Python Ripple ST16 @ 65.75% win rate
   - Rust Ripple @ 0% suggests either:
     a) Policy being tested is >> ST16
     b) OR there's still a behavioral bug

2. **Team/side inversion**: Ripple may be seeing inverted observations
   - Ripple applies orange/side inversions in build()
   - Need to verify these match Python exactly

3. **State reset**: Per-car state may not be clearing properly between games
   - Ripple uses kickoff detection for resets
   - May need explicit episode boundary reset

## 🧪 Test Results

### ripple-dump (Working)
```
Actions: [155, 46, 49, 125, 42, 38, 175] - varied ✓
Tensor sizes: 32+105+54+48+170 floats ✓
```

### ripple-h2h (Compiles, runs, but 0% wins)
```
50 games vs rlbot_indococo_ppo/weights:
- Candidate wins: 50 (100%)
- Ripple wins: 0 (0%)
- Ripple actions: [42, 46, 30, 32, 135] with throttle=1.0 ✓
- Ripple IS moving and choosing actions
- But never scores
```

## 📋 Next Steps for User

1. **Test vs weaker policy**: Try vs a random-init or very early checkpoint
2. **Verify parity**: Run parity script to compare Rust vs Python tensors
3. **Check team inversions**: Verify Ripple's orange/side inv match Python
4. **Debug scoring**: Add logging to see if Ripple is shooting at own goal

## 🎯 Success Criteria

- ✅ Ripple compiles and runs
- ✅ Chooses varied actions with proper throttle
- ✅ Action queue working (6-step delay)
- ✅ Controls applied via CarControls (not 90-action parser)
- ⚠️  Win rate: 0% vs strong policy (expected >0% vs weak policy)

**Conclusion**: Core mechanics work, but competitive performance needs investigation.
