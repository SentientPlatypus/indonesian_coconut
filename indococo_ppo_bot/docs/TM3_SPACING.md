# TM3 team spacing analysis (Nov 16, 2026)

Trigger: user tested TM3 in RLBot: "really bad spacing between the cars ...
not working together". This measures spacing in simulation for TM3 final
(exp_TM3 1794807177), TM2 final (exp_TM2 1794776924) and TM4 at 30.21B
(exp_TM4 1794857969, copied before use), against Nexto, Necto and self-play,
with Nexto's / Necto's own teams as the reference.

## Method

- `TRAJ_DUMP=<prefix>` (new, env-gated, src/bin/train.rs `traj_line`): the
  `h2h` and `nexto-h2h` / `necto-h2h` workers write one JSON line per policy
  step (8 ticks): ball and every car (team, pos, vel, boost, demoed,
  cumulative touches). Nothing changes when it is unset.
- `tools/team_spacing_run.sh TAG CKPT CORES GAMES`: 2v2 and 3v3 x {nexto,
  necto, self}, 200 games each, nice 19, 4 pinned cores, 1 worker thread,
  270-wide pred obs (235-wide TM2 widened on load as in the evals).
- `tools/team_spacing.py LABEL:SIDE:GLOB ...`: the metrics (definitions in
  its docstring). Spacing stats skip the kickoff phase (before the first
  touch) and steps with a demoed teammate.
- Harness caveats: games are first-goal-wins from kickoff (~30 s each);
  Nexto / Necto use delay 6 / 7, scripted kickoffs and spawn jitter; our
  side is greedy; self-play `h2h` has 3% random actions and no jitter.
  Nexto(all) / Necto(all) pool the bot's team across the TM2 / TM3 / TM4
  games (600 games).

## Results

Win rates (decisive, our side): TM3 vs Nexto 0.430 (2v2) / 0.195 (3v3),
vs Necto 0.815 / 0.740; TM2 vs Nexto 0.385 / 0.225; TM4 vs Nexto 0.500 / 0.235.

#### 2v2

| metric | TM3vNexto | TM3vNecto | TM3self | TM2vNexto | TM2self | TM4vNexto | TM4self | Nexto(all) | Necto(all) |
|---|---|---|---|---|---|---|---|---|---|
| games | 200 | 200 | 400 | 200 | 400 | 200 | 400 | 600 | 600 |
| wins | 86 | 163 | 200 | 77 | 200 | 100 | 200 | 337 | 105 |
| mate_dist_p10 | 426 | 387 | 472 | 419 | 441 | 459 | 493 | 1036 | 588 |
| mate_dist_p50 | 1951 | 1688 | 2073 | 1892 | 1952 | 1923 | 2048 | 3044 | 2159 |
| mate_dist_mean | 2627 | 2328 | 2758 | 2611 | 2653 | 2618 | 2688 | 3403 | 2560 |
| pair<800 | 0.213 | 0.244 | 0.193 | 0.219 | 0.213 | 0.198 | 0.186 | 0.061 | 0.153 |
| pair<1000 | 0.268 | 0.311 | 0.251 | 0.280 | 0.271 | 0.258 | 0.244 | 0.093 | 0.206 |
| two_near_ball | 0.276 | 0.325 | 0.258 | 0.277 | 0.274 | 0.275 | 0.249 | 0.117 | 0.209 |
| dc600 | 0.032 | 0.036 | 0.022 | 0.034 | 0.025 | 0.028 | 0.022 | 0.003 | 0.006 |
| dc500 | 0.041 | 0.048 | 0.030 | 0.043 | 0.035 | 0.037 | 0.027 | 0.005 | 0.013 |
| goalside_any | 0.842 | 0.861 | 0.879 | 0.842 | 0.878 | 0.859 | 0.883 | 0.921 | 0.896 |
| last_back_deep | 0.391 | 0.374 | 0.457 | 0.400 | 0.441 | 0.401 | 0.456 | 0.563 | 0.468 |
| all_ahead | 0.158 | 0.139 | 0.121 | 0.158 | 0.122 | 0.141 | 0.117 | 0.079 | 0.104 |
| first_swaps_per_min | 10.392 | 9.112 | 10.616 | 10.718 | 10.285 | 10.609 | 10.699 | 11.095 | 11.696 |
| role_run_s | 4.713 | 4.937 | 4.644 | 4.508 | 4.709 | 4.673 | 4.551 | 4.445 | 4.073 |
| kick_2plus | 0.215 | 0.310 | 0.220 | 0.255 | 0.282 | 0.165 | 0.158 | 0.378 | 0.047 |
| touch_top_share | 0.783 | 0.798 | 0.732 | 0.771 | 0.740 | 0.762 | 0.733 | 0.739 | 0.757 |
| passes_per_min | 1.555 | 1.688 | 1.612 | 1.568 | 1.652 | 1.740 | 1.646 | 1.595 | 1.103 |
| boost_steal | 0.190 | 0.336 | 0.186 | 0.178 | 0.194 | 0.239 | 0.172 | 0.195 | 0.034 |
| r1_ball | 1051 | 911 | 1122 | 1066 | 1103 | 1045 | 1094 | 1319 | 1337 |
| r2_ball | 2875 | 2657 | 3103 | 2914 | 3032 | 2906 | 3022 | 3704 | 3032 |
| r2_goal | 4964 | 5420 | 4602 | 4809 | 4580 | 5000 | 4620 | 4223 | 3806 |

#### 3v3

| metric | TM3vNexto | TM3vNecto | TM3self | TM2vNexto | TM2self | TM4vNexto | TM4self | Nexto(all) | Necto(all) |
|---|---|---|---|---|---|---|---|---|---|
| games | 200 | 200 | 400 | 200 | 400 | 200 | 400 | 600 | 600 |
| wins | 39 | 148 | 200 | 45 | 200 | 47 | 200 | 469 | 145 |
| mate_dist_p10 | 534 | 499 | 576 | 505 | 570 | 569 | 597 | 1189 | 656 |
| mate_dist_p50 | 2422 | 2303 | 2476 | 2288 | 2515 | 2514 | 2570 | 3393 | 2279 |
| mate_dist_mean | 3112 | 2980 | 3122 | 3008 | 3122 | 3162 | 3181 | 3702 | 2689 |
| pair<800 | 0.392 | 0.422 | 0.371 | 0.412 | 0.374 | 0.375 | 0.352 | 0.130 | 0.325 |
| pair<1000 | 0.482 | 0.517 | 0.462 | 0.498 | 0.466 | 0.462 | 0.443 | 0.196 | 0.422 |
| two_near_ball | 0.386 | 0.434 | 0.379 | 0.402 | 0.379 | 0.373 | 0.368 | 0.207 | 0.327 |
| dc600 | 0.038 | 0.050 | 0.035 | 0.043 | 0.038 | 0.038 | 0.036 | 0.005 | 0.012 |
| dc500 | 0.042 | 0.060 | 0.041 | 0.051 | 0.044 | 0.041 | 0.041 | 0.007 | 0.024 |
| goalside_any | 0.896 | 0.937 | 0.933 | 0.908 | 0.932 | 0.917 | 0.947 | 0.967 | 0.952 |
| last_back_deep | 0.555 | 0.632 | 0.632 | 0.563 | 0.636 | 0.587 | 0.660 | 0.729 | 0.618 |
| all_ahead | 0.104 | 0.063 | 0.067 | 0.092 | 0.068 | 0.083 | 0.053 | 0.033 | 0.048 |
| first_swaps_per_min | 18.430 | 16.055 | 16.862 | 18.304 | 17.340 | 18.758 | 17.142 | 18.107 | 18.511 |
| role_run_s | 2.956 | 3.348 | 3.226 | 2.993 | 3.046 | 2.941 | 3.170 | 3.021 | 2.943 |
| kick_2plus | 0.300 | 0.445 | 0.458 | 0.300 | 0.455 | 0.195 | 0.295 | 0.635 | 0.157 |
| touch_top_share | 0.643 | 0.627 | 0.626 | 0.617 | 0.610 | 0.612 | 0.594 | 0.593 | 0.596 |
| passes_per_min | 2.063 | 2.989 | 2.157 | 2.116 | 2.544 | 1.868 | 2.281 | 2.806 | 1.607 |
| boost_steal | 0.142 | 0.294 | 0.116 | 0.129 | 0.118 | 0.128 | 0.118 | 0.221 | 0.037 |
| r1_ball | 1057 | 944 | 1060 | 1064 | 1066 | 1089 | 1058 | 1207 | 1224 |
| r2_ball | 2178 | 1978 | 2187 | 2100 | 2225 | 2206 | 2235 | 2624 | 2289 |
| r3_ball | 4220 | 4139 | 4342 | 4137 | 4328 | 4299 | 4393 | 4942 | 3931 |
| r2_goal | 4884 | 5924 | 5023 | 4843 | 5007 | 4866 | 4964 | 5044 | 3728 |
| r3_goal | 4279 | 4773 | 4098 | 4155 | 4095 | 4210 | 3979 | 4112 | 2975 |

## Diagnosis

1. Clumping is real and large, and it is the same in TM2, TM3 and TM4.
   Teammates are within 800 uu of each other 19-24% of live play in 2v2 and
   35-42% in 3v3, against Nexto's 6% and 13%. The 10th-percentile teammate
   distance is ~390-600 uu (Nexto ~1040-1190). Two or more of our cars are
   within 1500 uu of the ball 25-33% (2v2) / 37-43% (3v3) of the time vs
   Nexto 12% / 21%. Double commits (repo test: 2 within 600 uu, both
   closing >= 300 uu/s) are 2-5% of steps vs Nexto 0.3-0.5%, ~10x.
2. The support car plays too close and too high. Second man is ~2.7-3.1k uu
   from the ball (Nexto 3.7k) in 2v2, and in 3v3 the third man is ~4.1-4.4k
   (Nexto 4.9k). A deep last man (goalside and 1500+ uu behind the ball) is
   there 37-46% of the time in 2v2 (Nexto 56%) and 56-66% in 3v3 (Nexto 73%).
   All our cars are ahead of the ball 12-16% of 2v2 steps (Nexto 8%).
3. It is not a kickoff or a role-identity problem. Two cars near the ball at
   the first kickoff touch: 16-31% (2v2), 20-46% (3v3), below Nexto's
   scripted kickoffs (38% / 64%). First-man swaps per minute (~10 in 2v2,
   ~17-19 in 3v3) and first-man stint length (~4.5 s / ~3 s) match Nexto.
   Passes per minute match Nexto in 2v2 (1.6) and are lower in 3v3 (2.1 vs
   2.8 vs Nexto).
4. Longer training is not fixing it: TM2 -> TM3 -> TM4 (27.3B -> 29.0B ->
   30.2B) moved 2v2 pair<800 0.219 -> 0.213 -> 0.198 and 3v3 0.412 -> 0.392
   -> 0.375 (vs Nexto).
5. Obs is not the cause. The team-ext obs has a closest-on-my-team flag, a
   distance-to-ball rank for self and every slot, and a kickoff-taker flag
   (ties broken toward the left car in the team frame). Orange's view is
   rotated 180 degrees (x and y both flipped) like the base obs; there is no
   per-car left/right mirror, so kickoff roles are not symmetric-ambiguous.

Reward side (resolved tm_rewards.toml chain st.toml -> st36 -> tm1 -> tm1_fr -> tm2):

- Nothing rewards spacing. `teammate_proximity` is 0 (archive: weight 12,
  radius 800, everyone penalized -> 3v3 avoided the ball). `double_commit`
  is 2 per step and only fires within 600 uu while both close fast, which is
  a small slice of the clumping (most is 600-1500 uu).
- Ball-centric terms dominate and are not closest-gated: touch 6, possession
  68, ball_travel 10, vel_ball_to_goal 12, goal_prob 40, aerial_distance 38,
  airdribble / airdribble_seq 35, flick 22, pressure_flick 36, high_ball 18,
  range_carry 28, clear_path 42; and individual (not team-shared):
  flip_reset 10, flip_reset_use 30, musty 40, air_double_touch 20. Only
  speed_to_ball (1), face_ball (1) and shadow_space (18) are closest-gated.
  With spirit 0.9 a car keeps 55% (2v2) / 40% (3v3) of its own shared term
  versus 45% / 30% for a teammate's, so being the one on the ball still pays
  most, and the individual mechanic terms pay only the toucher.
- `concede_dist` 300 is individual and charges every car
  -(1 - exp(-d / 2300)) by its distance to the ball when a goal goes in
  (~-250 at 4000 uu). It pulls every car toward the ball. Necto uses the
  same term and Necto also clumps more than Nexto (pair<800 0.15 / 0.33).
- Positioning rewards exist only for the last-back car (shadow_box 30 with
  team_last_back, recover 70). Nothing places the second man.
- Pass 40 needs >= 800 uu of ball travel, so it does not directly reward
  standing close, but it gives no reason to stand far either.

## Ranked TM5 fixes

1. Banded teammate spacing, individual, first man exempt. Per car (not the
   team's closest to the ball, not demoed, not during kickoff): nearest
   teammate distance d, -1 at d <= 600 ramping to 0 at 1400, plus -0.25 when
   d > 6000 (too far). Weight 3 per step, ramped 0 -> 3 over the first 300M.
   The first-man exemption is the change from the archive penalty that made
   3v3 avoid the ball. Keep it below what a real commit pays (Zealan), and
   watch r1_ball and touches per minute for the passive-defense failure
   (Kaiyotech). Target: pair<800 near Nexto (0.06 / 0.13).
2. Gate ball-commit terms against double commits. Add touch, aerial_distance,
   airdribble, airdribble_seq, flick, pressure_flick, high_ball,
   air_double_touch to `no_double_commit_gate`, and raise `double_commit` to
   5 with radius 1000 / min_approach 200 so it covers the 600-1500 uu band.
3. Second-man / last-man positioning. A goalside bonus for the team's
   farthest-from-ball car: +1 per step when goalside and 1500-4500 uu behind
   the ball, weight 2, last_back_gate. In 3v3, the rank-1 car gets +0.5 per
   step for 2000-3500 uu from the ball. Target last_back_deep >= 0.55 (2v2).
4. Rework concede_dist. Charge only the team's last-back car, or measure
   distance to the own goal instead of the ball, or cut it 300 -> 100.
5. team_spirit 0.9 -> 1.0, together with 1 (Saturn / Annie recipe). On its
   own it barely changes the own-vs-teammate split.
6. Mixed Nexto slots as teammates (Nexto + learner on one team) in 10-20% of
   team games. This needs collector code changes. TM4's Nexto opponents
   (7%) did not change spacing.
7. Longer training alone, last: TM2 -> TM4 (+3B) moved spacing ~1-2 points.

Kickoffs need no fix (no kickoff double-commit excess vs Nexto).
