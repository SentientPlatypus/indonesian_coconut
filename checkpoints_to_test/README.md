# checkpoints_to_test/

Policy weights worth testing, one `<name>.bpk` per checkpoint. New candidates get added here the same way.

## How to test one

1. Keep a single RLBot bot folder: either [`rlbot_indococo_ppo/`](../rlbot_indococo_ppo/) on this branch, or the folder from the [adb-832m release](https://github.com/SentientPlatypus/indonesian_coconut/releases/tag/adb-832m).
2. Copy the `.bpk` you want over `<bot>/weights/actor.bpk`, e.g. `cp checkpoints_to_test/adb_832M.bpk rlbot_indococo_ppo/weights/actor.bpk`.
3. Restart the bot (or the match). Weights load only at startup.

**Compatibility:** a `.bpk` only works with binaries built for the same obs, action and network layout. Every file here uses `V2DefaultObs`, the indococo action set and the current actor network. The `indococo-rlbot`/`.exe` binaries in `rlbot_indococo_ppo/` and in the release are byte-identical and run all of them. If the obs, actions or network change, rebuild the binaries and say so here.

## Checkpoints

Headless evals use seed 424242, the air-time clamp on (as deployed), a fresh engine per game and mirrored seeds. Pool = 400 games per opponent, win %.

### st3_5772M.bpk (current best)

| | |
|---|---|
| Lineage | ST3 strength run (ST2 recipe + entropy 0.008, LR 4e-5 cosine to 0.25x), chunk `1793809757` (5771.6M steps), warm-started from ST2 5511.6M. 106-input obs (`INDOCOCO_OBS_EXT=1`) |
| H2H | **53.4%** vs ST2 5511.6M (534-466), **69.1%** vs FR4, over 1000 games each |
| Pool | **91.17** on pool400 (ST2 89.12): 1v1.pt 77.75, 430M 84.5, Nexto 91.5, Necto 98.0, Element 98.25, Immortal 97.0 |
| Botpack5 | 61.5 avg (ST2 61.58): Ripple 56.75, Kira 60.75, Dice 56.25, Chronos 74.25, Galactus 21.25, Seer 99.75. Auto-promoted |
| Galactus | 20.4% over 1000 games (ST2 20.8%) |
| sha256 | `bd72a8a7293f88f463b536242cce51873c1af0a4dd8f1541b2f97d14e0d23b75` |

### st2_5512M.bpk

| | |
|---|---|
| Lineage | ST2 strength run (ST1 recipe + gamma 0.997, 200k/iter, mb 100k), chunk `1793792743` (5511.6M steps), warm-started from ST1 5051.6M. 106-input obs (`INDOCOCO_OBS_EXT=1`) |
| H2H | **68.2%** vs FR8-final (682-318), **64.4%** vs FR4, over 1000 games each |
| Pool | **89.12** on pool400 (FR8-final 86.95): 1v1.pt 75.5, 430M 75.5, Nexto 91.25, Necto 96.75, Element 99.0, Immortal 96.75 |
| Botpack5 | **61.58** avg (FR8-final 58.0): Ripple 58.75, Kira 61.5, Dice 54.75, Chronos 72.25, Galactus 22.25, Seer 100. Auto-promoted |
| sha256 | `1fc134ee15bae7c810a07b6c62095a2f1b0d488a4d0caec366388ecfebc37bc3` |

### st1_5052M.bpk

| | |
|---|---|
| Lineage | ST1 strength run (no reset rewards/drills, no KL anchor, league of FR4/FR7/FR8/FR9/FR9b/1v1.pt + self), chunk `1793773641` (5051.6M steps), warm-started from FR8-final. 106-input obs (`INDOCOCO_OBS_EXT=1`) |
| H2H | **62.2%** vs FR8-final (622-378), **61.6%** vs FR4, over 1000 games each |
| Pool | **86.62** on pool400 (FR8-final 86.95; Nexto 85.8 vs 88.5) |
| Botpack5 | 56.25 avg (FR8-final 58.0; Ripple 48.8, Kira 51.0). Not auto-promoted (rule allows at most a 1-point botpack drop) |
| sha256 | `eec1a3b2a162138a26c21a341adc3e5cab787dd1856e0f3e07b8993db9fc4297` |

### aud.pt
The AUD 1132M policy (same weights as `aud_1132M.bpk`) converted to the Python `DiscreteFF` state_dict format with `indococo_ppo_bot/tools/bpk_to_pt/`. It's byte-identical to `rlbot_submission/policies/1v1.pt` on master (sha256 `15e16300…53f68`). To test it in the Python bot: `cp checkpoints_to_test/aud.pt rlbot_submission/policies/1v1.pt`.

### fr6_2632M.bpk

| | |
|---|---|
| Lineage | FR6 arm (FR4-final + behaviour cloning of a scripted reset expert, then 1.2B steps of PPO in 50M chunks, each followed by a BC refresh anchored to FR4-final), 2631.6M steps (trainer dir `1793540902`) |
| H2H | **48.0%** vs FR4-final (480-520), **47.6%** vs AUD-final (476-524), over 1000 games each |
| Pool | **85.0**: 1v1.pt 65.8, 430M 69.3, Nexto 85.8, Necto 96.5, Element 98.3, Immortal 94.5 |
| Flip resets | First checkpoint here that *uses* resets: on `rates_fr3.toml` (300k steps) 9 resets, **4 uses**, 0 goals after a reset. All 4 uses came in `reset_curriculum` start states; kickoff-started play had 1 reset and 0 uses. Curriculum level-0 success 29.8% greedy (FR4: 1.9%), ~94% of resets used. |
| sha256 | `4fe4736eeb0a92dce7c7347945d9e8b693856427720ec3a435c2b885df9cefb5` |

### fr4_1432M.bpk

| | |
|---|---|
| Lineage | FR4 arm (reset curriculum on the AUD air rewards), 1431.6M steps (trainer dir `1793481832`), warm-started from AUD-final |
| H2H | **63.3%** vs ADB-best (633-367), **53.2%** vs AUD-final (532-468), over 1000 games each |
| Pool | **87.1**: 1v1.pt 68.0, 430M 71.0, Nexto 89.8, Necto 98.0, Element 99.5, Immortal 96.5 |
| Flip resets | 0.003 resets/min on `rates_fr3.toml`, 0 uses. The curriculum never left level 0. |
| sha256 | `a1c62359ea196833dc5915a0b96c8bf42dbc684bf8c04386ab34047c5d7c189d` |

### aud_1132M.bpk

| | |
|---|---|
| Lineage | AUD reward-audit arm (air rewards retuned: airdribble_seq 50, airdribble 50, aerial_distance 38, air_touch_height 0, goalward-only reworks), 1131.6M steps (trainer dir `1793475982`), warm-started from ADB `ckpt_831600000` |
| H2H | **64.6%** vs ADB-best (646-354 over 1000 games) |
| Pool | **87.0**: 1v1.pt 72.5, 430M 69.5, Nexto 87.2, Necto 97.5, Element 99.0, Immortal 96.5 |
| Flip resets | 0.111 resets/min with the fixed detector, 0 reset uses, none from the air_reset setter |
| sha256 | `bb48b9b1ff0594aaf0be1d70d8ed9d41b2feebdc77802e9a34cbc35027ef4773` |

### fr2_1132M.bpk

| | |
|---|---|
| Lineage | FR2 arm (flip-reset rewards with ramped air_reset/air_near_ball setters), 1131.6M steps (trainer dir `1793462035`), warm-started from ADB `ckpt_831600000` |
| H2H | **57.2%** vs ADB-best (572-428 over 1000 games) |
| Pool | **83.5**: 1v1.pt 66.8, 430M 70.0, Nexto 80.8, Necto 92.5, Element 96.2, Immortal 94.5 |
| Flip resets | 0.051 resets/min with the fixed detector (X0 0.064, ADB 0.078), 0 reset uses, none from the air_reset setter. The arm did not teach resets. |
| sha256 | `da775c2dcb9826faf6a7db0570dc9e004135e50ef85155c14bc3f7e349ba3178` |

### adb_832M.bpk

| | |
|---|---|
| Lineage | ADB arm (air-dribble bump + air_touch_height + victim-gated demo), `ckpt_831600000` (831.6M steps), warm-started from 430M |
| H2H | **63.6%** vs 430M (636-364), **52.1%** vs X0-best (521-479), over 1000 games each |
| Pool | **84.7**: 1v1.pt 65.2, 430M 65.0, Nexto 85.5, Necto 95.8, Element 99.0, Immortal 97.5 |
| sha256 | `dba1af19156079204b90e7d10c22b9a2ebd499ff0a22905789c4818067c61b0e` |

### 430M.bpk

| | |
|---|---|
| Lineage | e8b2_1v1 run from 1v1.pt imported into Burn, 430.8M steps (`policy_430800000`) |
| H2H | **60.2%** vs 1v1.pt (602-398 over 1000 games) |
| Pool | **79.0** (aggregate only; per-opponent numbers were not recorded). 1v1.pt scored 77.5. |
| sha256 | `5ea50c54fe40d760677483262f901d35c96dc12720a50fcd47607c83efa2f121` |
