# Ripple Observation Layout

Source: Rolv-Arild/Ripple-RLBot @ 7ca4273 (Initial commit)
File: `src/obs.py`

## Main Player Tensor (32 elements)

From lines 212-237 in obs.py:

```
[0:3]   position                    (line 213)
[3:6]   linear_velocity             (line 214)
[6:9]   angular_velocity            (line 215)
[9:12]  forward()                   (line 216)
[12:15] up()                        (line 217)
[15]    team (2 * team_num - 1)     (line 218)
[16]    demo_timer                  (line 220)
[17]    on_ground                   (line 221)
[18]    has_jump                    (line 222)
[19]    has_flip                    (line 223)
[20]    boost_amount (min(., 2))    (line 224)
[21]    boost_times[i]              (line 225)
[22]    jump_times[i]               (line 226)
[23]    air_times[i]                (line 227)
[24]    flip_times[i]               (line 228)
[25]    handbrakes[i]               (line 229)
[26:28] flip_dirs[i] (2 floats)     (line 230)
[28]    is_reset                    (line 232)
[29]    is_kickoff (becomes flip indicator) (line 234)
[30]    is_flipped (set at line 273) (line 236 commented)
[31]    is_swapped (set at line 264) (line 237 commented)
```

**CRITICAL: NO RIGHT VECTOR!** Python only stores forward (x-axis) and up (z-axis), not the right (y-axis) vector.

Lines 246-273 apply coordinate transforms:
- Line 249: Swap player i with player 0 (current player to index 0)
- Line 250-251: Make positions/velocities relative, rotate into car frame
- Line 259-264: If orange team, invert certain axes
- Line 266-273: If on left side (x < 0), mirror to right side

## Other Players Tensor (5 players × 21 elements)

Line 281: `other_players = players[:, 1:, :21]` - Only first 21 fields, no timers/flags

```
[0:3]   position (relative, rotated)
[3:6]   linear_velocity (relative, rotated)
[6:9]   angular_velocity (rotated)
[9:12]  forward() (rotated)
[12:15] up() (rotated)
[15]    team
[16]    demo_timer
[17]    on_ground
[18]    has_jump
[19]    has_flip
[20]    boost_amount
```

## Ball Tensor (6 time slices × 9 elements)

Lines 186, 207: ball_predictions array

```
[0:3]   position (relative, rotated)
[3:6]   linear_velocity (relative, rotated)
[6:9]   angular_velocity (rotated)
```

## Actions Tensor (7 actions × 8 elements)

Line 238: `actions[i] = self.action_queue[:, i, :]`
Action queue is populated externally, 8-element action vectors

## Boosts Tensor (34 pads × 5 elements)

Lines 240-242:

```
[0:3]   position (BOOST_LOCATIONS, relative, rotated)
[3]     is_big (0.12 + 0.88 * (z > 72))
[4]     timer / 10
```

## Normalization

Line 66: `NORM = np.array([2300] * 6 + [6] * 3 + [1] * 6 + [1, 3, 1, 1, 1, 1, 12, 24, 150, 120, 1, 1, 1, 1, 1, 1, 1])`

```
[0:6]   pos+vel: / 2300
[6:9]   ang_vel: / 6
[9:15]  fwd+up: / 1
[15]    team: / 1
[16]    demo: / 3
[17:20] on_gnd, has_j, has_f: / 1
[20]    boost: / 1
[21]    boost_t: / 12
[22]    jump_t: / 24
[23]    air_t: / 150
[24]    flip_t: / 120
[25]    handbrake: / 1
[26:32] flip_dir, is_reset, is_kickoff, is_flipped, is_swapped: / 1
```

## Key Issues

1. **NO RIGHT VECTOR**: Python only stores forward and up, not right. The test harness expects [12-14] to be "rgt" but Python never fills it!
2. **up() returns full 3D vector**: Not a right vector
3. **Coordinate transforms**: Lines 250-273 apply rotation, team inversion, side mirroring
