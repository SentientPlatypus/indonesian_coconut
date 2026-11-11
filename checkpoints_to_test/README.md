# checkpoints_to_test/

Policy weights worth testing, one `<name>.bpk` per checkpoint. New candidates get added here the same way.

## How to test one

1. Keep a single RLBot bot folder: either [`rlbot_indococo_ppo/`](../rlbot_indococo_ppo/) on this branch, or the folder from the [adb-832m release](https://github.com/SentientPlatypus/indonesian_coconut/releases/tag/adb-832m).
2. Copy the `.bpk` you want over `<bot>/weights/actor.bpk`, e.g. `cp checkpoints_to_test/adb_832M.bpk rlbot_indococo_ppo/weights/actor.bpk`.
3. Restart the bot (or the match). Weights load only at startup.

**Compatibility:** a `.bpk` only works with binaries built for the same obs, action and network layout. Every file here uses `V2DefaultObs`, the indococo action set and the current actor network. The `indococo-rlbot`/`.exe` binaries in `rlbot_indococo_ppo/` and in the release are byte-identical and run all of them. If the obs, actions or network change, rebuild the binaries and say so here.

## Checkpoints

Headless evals use seed 424242, the air-time clamp on (as deployed), a fresh engine per game and mirrored seeds. Pool = 400 games per opponent, win %.

### st25_20728M.bpk (current best)

| | |
|---|---|
| Lineage | ST25 (ST24 config = ST23 recipe + `ST20r_final` @ 2.0, OPPCH 0.70, 1 PPO epoch, binary `9d4b2a6`; demo_threat setter retuned to 0.15 of starts at 1800-3000 uu, normal play x0.875, aerial_defense 0.15), continued from ST24 19727.3M, selected chunk `1794413355` (20727.5M steps; training ended there). 106-input obs (`INDOCOCO_OBS_EXT=1`). **Manual promote**, user-approved 2026-11-11 12:16 PM ET. Auto gate failed only on FR4 (78.4 vs ST23 79.6, 1.2 below; tolerance 1.0) |
| H2H | **56.2%** vs ST23 (281-219), **78.4%** vs FR4 (ST23 79.6%), over 500 games each |
| Pool | **93.33** on pool400 (ST23 93.21): 1v1.pt 85.75, 430M 86.75, Nexto 92.5, Necto 99.25, Element 99.0, Immortal 96.74 |
| Botpack5 | **69.21** avg (ST23 68.67): Ripple 68.25, Kira 66.5, Dice 61.75, Chronos 78.75, Galactus 40.25, Seer 99.75 |
| Galactus | **40.8%** over 500 games (ST23 35.4%), best so far |
| Megabyte | **84.0%** over 500 games (report-only; ST23 83.2%) |
| Gen panel | min **0.408** / mean **0.8132** (ST23 0.354 / 0.8087) |
| sha256 | `db3f77b4cdbbec57c7f227e32fd5210c450fda250b042dfabe704d0989923ed8` |

### st23_19016M.bpk

| | |
|---|---|
| Lineage | ST23 (ST21 recipe + OPPCH 0.70 + 1 PPO epoch, ST21 setters, no `air_dribble_wall_challenge`, GAL_WEIGHT 6.429, MEGA_WEIGHT 2.0, Ripple toml 1.286, ENT 0.016), continued from ST21 18053.6M, keep-best chunk `1794367057` (19015.7M steps; training ended at 19053.8M). 106-input obs (`INDOCOCO_OBS_EXT=1`). **Auto promote**, user confirmed 2026-11-10 11:30 PM ET. All gates passed. The OPPCH bump and the 1-epoch switch are confounded, and the wall-challenge setter was also dropped |
| H2H | **56.0%** vs ST21 (280-220), **79.6%** vs FR4 (ST21 75.4%), over 500 games each |
| Pool | **93.21** on pool400 (ST21 92.75): 1v1.pt 86.5, 430M 85.0, Nexto 91.8, Necto 98.8, Element 99.0, Immortal 98.2 |
| Botpack5 | **68.67** avg (ST21 68.17): Ripple 67.25, Kira 68.5, Dice 65.25, Chronos 75.25, Galactus 36.25, Seer 99.5 |
| Galactus | **35.4%** over 500 games (ST21 35.6%) |
| Megabyte | **83.2%** over 500 games (report-only; ST21 81.6%) |
| Gen panel | min **0.354** / mean **0.8087** (ST21 0.356 / 0.8034) |
| sha256 | `b135fae887d0a0733c2ef712cccdd0737a979af54a69bbfdd4c2e0b5f4f81e57` |

### st21_18054M.bpk


| | |
|---|---|
| Lineage | ST21 (ST20r recipe + Galactus 25% / Ripple 5% / Megabyte 7.8%, GAL_WEIGHT 6.429, MEGA_WEIGHT 2.0, Ripple toml 1.286, OPPCH 0.60, ENT 0.016, gal4 setters), continued from ST20r 17642.6M, keep-best chunk `1794296692` (18053.6M steps; training ended at 18642.8M). 106-input obs (`INDOCOCO_OBS_EXT=1`). **Manual promote** 2026-11-10 08:57 ET (user). The auto gate failed only on bp5 68.17 vs ST20r 69.74 (about 1.6 short of the −1.0 tolerance). All other gates passed |
| H2H | **52.8%** vs ST20r (264-236), **75.4%** vs FR4 (ST20r 75.8%), over 500 games each |
| Pool | **92.75** on pool400 (ST20r 92.42): 1v1.pt 85.5, 430M 85.0, Nexto 92.5, Necto 98.25, Element 98.5, Immortal 96.75 |
| Botpack5 | **68.17** avg (ST20r 69.74): Ripple 71.75, Kira 66.5, Dice 57.25, Chronos 76.75, Galactus 37.0, Seer 99.75 |
| Galactus | **35.6%** over 500 games (ST20r 34.2%) |
| Megabyte | **81.6%** over 500 games (report-only; ST20r 79.2%) |
| Gen panel | min **0.356** / mean **0.8034** (ST20r 0.342 / 0.8107) |
| sha256 | `505b0249d4c233bdf8b40e46a5b4422065fdef79c5b8890ebde36b107bd6a9e5` |

### st20r_17643M.bpk


| | |
|---|---|
| Lineage | ST20r (repeat of the ST20 recipe: ST19 gal4 setters + Rust Ripple v1.1 at 5%, GAL_WEIGHT 5.143 ~20%, MEGA_WEIGHT 2.0, OPPCH 0.60, ENT 0.016), continued from ST20 16880.9M, chunk `1794276990` (17642.6M steps). 106-input obs (`INDOCOCO_OBS_EXT=1`). **Manual promote** 2026-11-10 00:16 ET (user). The auto gate was scored against ST20 (wrong baseline) and the formal h2h vs ST19r was skipped. A 500-game h2h vs ST19r finished just before and is report-only: 56.4% (282-218). Every gate vs ST19r passes |
| H2H | 56.4% vs ST19r (282-218, report-only), 58.4% vs ST20 (292-208), **75.8%** vs FR4 (ST19r 71.6%), over 500 games each |
| Pool | **92.42** on pool400 (ST19r 92.62): 1v1.pt 82.75, 430M 83.25, Nexto 94.25, Necto 98.5, Element 99.0, Immortal 96.75 |
| Botpack5 | **69.74** avg (ST19r 67.38): Ripple 72.5, Kira 67.5, Dice 67.7, Chronos 76.75, Galactus 34.25, Seer 99.75 |
| Galactus | **34.2%** over 500 games (ST19r 34.2%) |
| Megabyte | **79.2%** over 500 games (report-only; ST19r 81.0%) |
| Gen panel | min **0.342** / mean **0.811** (ST19r 0.342 / 0.800) |
| sha256 | `4711f09d65be97030088f49af76a7b8f9bfec53b69aea9f12f4eb5df77ef78a4` |

### st19r_15919M.bpk

| | |
|---|---|
| Lineage | ST19r (ST19 recipe continued from ST19 15307.4M: aerial_defense 0.15 + demo_threat 0.05, GAL_WEIGHT 4.5, MEGA_WEIGHT 2.0, OPPCH 0.60, ENT 0.016), chunk `1794217094` (15918.8M steps). 106-input obs (`INDOCOCO_OBS_EXT=1`). **Manual promote** 2026-11-09: auto gate failed on h2h vs ST16 (50.4%, needs ≥52) and FR4 (−3.0 vs −1.0 tol). Galactus, Megabyte, pool, botpack5 and the gen panel improved |
| H2H | 50.4% vs ST16 (252-248), **71.6%** vs FR4 (ST16 74.6%), over 500 games each |
| Pool | **92.62** on pool400 (ST16 92.04): 1v1.pt 82.5, 430M 83.75, Nexto 93.0, Necto 97.25, Element 99.75, Immortal 99.5 |
| Botpack5 | **67.38** avg (ST16 65.83): Ripple 65.0, Kira 67.5, Dice 60.5, Chronos 77.25, Galactus 34.0, Seer 100.0 |
| Galactus | **34.2%** over 500 games (ST16 25.0%) |
| Megabyte | **81.0%** over 500 games (report-only; ST16 71.6%) |
| Gen panel | min **0.342** / mean **0.800** (ST16 0.250 / 0.788) |
| sha256 | `5a2405ebf11dc72131ad964518366df3dad22e8487c8707f167c631845d6fe6a` |

### st10_10629M.bpk

| | |
|---|---|
| Lineage | ST10 generalization run (ST9 + ST7_final in EXTRA_OPPS + Megabyte@2.0 via indococo-train-mega; ENT 0.016, OPPCH 0.55, Galactus w2.5), chunk `1793988079` (10628.6M steps), warm-started from ST9 9628.4M. 106-input obs (`INDOCOCO_OBS_EXT=1`). **Manual promote** (user 2026-11-06 1:36 PM ET): auto gate failed only on FR4 −1.6 vs −1.0 tol; within ~2 pt noise on 500 games; all other metrics improved |
| H2H | **54.6%** vs ST9 9628.4M (273-227), **71.2%** vs FR4 (ST9 72.8%), over 500 games each |
| Pool | **92.29** on pool400 (ST9 91.67): 1v1.pt 82.25, 430M 82.0, Nexto 93.0, Necto 99.0, Element 98.75, Immortal 98.75 |
| Botpack5 | **66.75** avg (ST9 63.21): Ripple 62.75, Kira 72.75, Dice 64.5, Chronos 75.75, Galactus 25.5, Seer 99.25 |
| Galactus | **24.4%** over 500 games (ST9 21.8%) |
| Megabyte | **71.0%** over 500 games (report-only; ST8 Python ref ~64%) |
| Gen panel | min 24.4% (Galactus), mean 79.43% over pool400 + botpack5 + Galactus (ST9 min 21.8%, mean 77.5%) |
| sha256 | `eeaa50b734e60121def5ac2bb6c9abf1ac92fd7a40116d511ff7ba4d9680b933` |

### st9_9628M.bpk

| | |
|---|---|
| Lineage | ST9 generalization run (ST8 recipe + OPPCH 0.55; ENT 0.016, Galactus w2.5, ST4/ST5/ST6 frozen finals in EXTRA_OPPS), chunk `1793962509` (9628.4M steps), warm-started from ST8 8666.3M. 106-input obs (`INDOCOCO_OBS_EXT=1`) |
| H2H | **54.0%** vs ST8 8666.3M (270-230), **72.8%** vs FR4 (ST8 73.0%), over 500 games each |
| Pool | **91.67** on pool400 (ST8 91.58): 1v1.pt 77.25, 430M 82.5, Nexto 92.25, Necto 99.75, Element 99.25, Immortal 99.0 |
| Botpack5 | **63.21** avg (ST8 64.21): Ripple 61.0, Kira 61.0, Dice 62.75, Chronos 74.0, Galactus 21.0, Seer 99.5. Auto-promoted |
| Galactus | **21.8%** over 500 games (ST8 25.2%) |
| Gen panel | min 21.8% (Galactus), mean 77.5% over pool400 + botpack5 + Galactus (ST8 min 25.2%, mean 77.87%) |
| sha256 | `c68d16d6965ee46d7bc44532f680603beebbc790a507a7de81500866334f4e90` |

### st8_8666M.bpk



| | |
|---|---|
| Lineage | ST8 generalization run (ST7 + ST4/ST5/ST6 frozen finals in EXTRA_OPPS; ENT 0.016, OPPCH 0.50, Galactus w2.5), chunk `1793939483` (8666.3M steps), warm-started from ST7 best mid 7666.1M. 106-input obs (`INDOCOCO_OBS_EXT=1`) |
| H2H | **55.4%** vs ST6 6954.5M (277-223), **73.0%** vs FR4 (ST6 71.2%), over 500 games each |
| Pool | **91.58** on pool400 (ST6 92.21): 1v1.pt 79.25, 430M 83.25, Nexto 92.0, Necto 97.5, Element 98.75, Immortal 98.75 |
| Botpack5 | **64.21** avg (ST6 63.42): Ripple 58.0, Kira 61.75, Dice 62.75, Chronos 77.25, Galactus 25.5, Seer 100.0. Auto-promoted |
| Galactus | **25.2%** over 500 games (ST6 23.8%) |
| Gen panel | min 25.2% (Galactus), mean 77.87% over pool400 + botpack5 + Galactus (ST6 min 23.8%, mean 77.63%) |
| sha256 | `5b6a77a692934b30bf8379d04910bb388dac84f37b1dd59c7dfc9b43f6d03679` |

### st6_6955M.bpk

| | |
|---|---|
| Lineage | ST6 generalization run (ST5 broader league + opponent chance 0.50), chunk `1793887366` (6954.5M steps), warm-started from ST5 6192.8M. 106-input obs (`INDOCOCO_OBS_EXT=1`) |
| H2H | **56.2%** vs ST5 6192.8M, **71.2%** vs FR4 (ST5 69.2%), over 500 games each |
| Pool | **92.21** on pool400 (ST5 91.21): 1v1.pt 83.75, 430M 80.75, Nexto 92.75, Necto 98.25, Element 99.25, Immortal 98.5 |
| Botpack5 | 63.42 avg (ST5 63.46): Ripple 61.0, Kira 55.25, Dice 60.25, Chronos 78.5, Galactus 26.0, Seer 99.5. Auto-promoted |
| Galactus | 23.8% over 500 games (ST5 23.4% over 1000) |
| Gen panel | min 23.8% (Galactus), mean 77.6% over pool400 + botpack5 + Galactus (ST5 min 23.4%, mean 77.3%) |
| sha256 | `63aedce21fc80e1ce7891da4404d3a747563a5f3a2bdf57143a953ba3cb7b199` |

### st5_6193M.bpk

| | |
|---|---|
| Lineage | ST5 generalization run (ST4 recipe + broader league: 7 extra frozen checkpoints X0/ADB/AUD/FR6/FR8c/FR9c/ST3 at w1.0, Galactus w2.5), chunk `1793851923` (6192.8M steps), warm-started from ST4 6032.3M. 106-input obs (`INDOCOCO_OBS_EXT=1`) |
| H2H | **57.4%** vs ST3 5771.6M (574-426), **69.2%** vs FR4, over 1000 games each |
| Pool | **91.21** on pool400 (ST3 91.17): 1v1.pt 80.25, 430M 82.0, Nexto 89.5, Necto 98.5, Element 99.25, Immortal 97.75 |
| Botpack5 | 63.46 avg (ST3 61.5): Ripple 65.25, Kira 61.0, Dice 62.25, Chronos 68.5, Galactus 23.75, Seer 100.0. Auto-promoted |
| Galactus | 23.4% over 1000 games (ST3 20.4%) |
| Gen panel | min 23.4% (Galactus), mean 77.3% over pool400 + botpack5 + Galactus (ST3 min 20.4%, mean 76.3%) |
| sha256 | `78bc5a40cbd8e3878abe613579a1462f66cf5c9bf55b08d1d1315c223fd61fc5` |

### st3_5772M.bpk

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
