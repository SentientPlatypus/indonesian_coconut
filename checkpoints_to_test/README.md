# checkpoints_to_test/

Policy weights worth testing, one `<name>.bpk` per checkpoint. New candidates get added here the same way.

## How to test one

1. Keep a single RLBot bot folder: either [`rlbot_indococo_ppo/`](../rlbot_indococo_ppo/) on this branch, or the folder from the [adb-832m release](https://github.com/SentientPlatypus/indonesian_coconut/releases/tag/adb-832m).
2. In `rlbot_indococo_ppo/`, copy the `.bpk` you want to `weights/<name>/actor.bpk` and run the bot with `--1v1 <name>` (1v1) or `--team <name>` (2v2/3v3); see [its README](../rlbot_indococo_ppo/README.md#choosing-the-model-st25-or-tm2-in-1v1). ST25 (`weights/st25`) and TM2 (`weights/tm2`) are already there: `bot.toml` plays ST25 in 1v1, `tm2.bot.toml` plays TM2 in 1v1, both play TM2 in 2v2/3v3. For the adb-832m release folder, copy over `<bot>/weights/actor.bpk` instead.
3. Restart the bot (or the match). Weights load only at startup.

**Compatibility:** a `.bpk` only works with binaries built for the same obs, action and network layout. Every file here uses `V2DefaultObs`, the indococo action set and the current actor network. The `indococo-rlbot`/`.exe` binaries in `rlbot_indococo_ppo/` pick the obs from the actor's input width (92, 106, team-ext 235, team-ext + prediction 270), so they run all of them; `tm*` files are team models that also play 1v1. If the obs, actions or network change, rebuild the binaries and say so here.

## Checkpoints

Headless evals use seed 424242, the air-time clamp on (as deployed), a fresh engine per game and mirrored seeds. Pool = 400 games per opponent, win %.

**Current references:** 1v1 BEST is ST25 (`st25_20728M.bpk` below). **team_best** is the TM9 final (`tm9_52957M.bpk`, user-approved 2026-11-22 7:45 PM ET, America/New_York). The RLBot pack still ships the TM2 final (27275M) as the default 2v2/3v3 model (`weights/tm2`); changing that is a separate pack decision. Every `*.bpk` in this folder should have a heading in this file.

### tm9_52957M.bpk (current team_best)

| | |
|---|---|
| Lineage | TM9 final (3v3 kickoff same-pad / one-corner rewards, `tm9.toml` / `train_tm9.toml`, 270-wide pred obs), last checkpoint at END, `exp_TM9/checkpoints/1795388102` (52957.1M steps). Warm start for TM10. **Promoted** to team_best, user-approved 2026-11-22 7:45 PM ET (America/New_York). |
| H2H | vs TM7 pin (400 g): 2v2 **57.5%** (230-170), 3v3 **65.5%** (262-138). vs TM8 final (400 g): 2v2 **51.75%** (207-193), 3v3 **47.75%** (191-209). |
| Nexto / Necto | Nexto 2v2 **60.3%** (181-119), 3v3 **48.7%** (146-154); Necto 2v2 **95.7%**, 3v3 **92.0%** (300 g each). |
| sha256 | `848d7fd295321b80a18db554fc3ee25f1f5836de5c79ca07d87919bd0f602021` |

### tm8_47957M.bpk

| | |
|---|---|
| Lineage | TM8 final (rotation cover, kickoff roles, open-net finishing, PFSP; 270-wide pred obs), `exp_TM8/checkpoints/1795278408` (47957.0M steps). Was team_best after user approval 2026-11-21 2:22 PM ET; superseded by TM9. Shipped, then promoted. |
| H2H | vs TM7 final (400 g): 2v2 **58.5%** (234-166), 3v3 **55.25%** (221-179). vs TM6 final (400 g): 2v2 **62.0%**, 3v3 **60.75%**. |
| Nexto / Necto | Nexto 2v2 **60.3%**, 3v3 **50.0%**; Necto 2v2 **92.7%**, 3v3 **91.3%** (300 g each). |
| sha256 | `6a12762fb75cce03d35d99b68f10fdb960d7d36f650c807270504b08d7af2ae7` |

### tm8_45582M.bpk

| | |
|---|---|
| Lineage | TM8 mid-run at 45582.2M (270-wide pred obs). **Not** the TM8 final and **not promoted**. |
| H2H (mid-run) | vs TM6: 2v2 **62.0%**, 3v3 **61.3%**; 1v1 vs ST25 **61.5%**; Nexto 2v2/3v3 **60.5%** / **53.9%**. |
| sha256 | `e790cfdb7ab85a21bdf95ce4113690b0a5388faa29a13ea6031c67d6d070bd37` |

### tm7_42957M.bpk

| | |
|---|---|
| Lineage | TM7 final (close + far spacing, pass separation, role-concede; 270-wide pred obs), 42957M steps, `exp_TM7/checkpoints/1795170349`. Was team_best; still the **TM10 BT pin**. Promoted at the time, later superseded by TM8/TM9. |
| sha256 | `304d04fbb26c2da45a72f484ae36b3d1b102b678d8c363d783689cc9784d3b2f` |

### tm6_37957M.bpk

| | |
|---|---|
| Lineage | TM6 final (role-based rewards; 270-wide pred obs), 37957M steps, `exp_TM6/checkpoints/1795044069`. Was team_best; later superseded. Promoted at the time. |
| sha256 | `b1436db0f326fdbcab8a6c4e23962e14d80acf2fdb7110bf54c14885a1147647` |

### tm5r_32957M.bpk

| | |
|---|---|
| Lineage | TM5r latest at the TM6 cut-over (stronger spacing, second-commit gate; 270-wide pred obs), 32957M steps. Cut-over export, **not** a final promote. |
| sha256 | `59c89fe5f7185514ba7f83ca00220cdd9e47fb09a8ccd63cb516f8c0b48c6532` |

### tm5_31434M.bpk

| | |
|---|---|
| Lineage | TM5 latest at the TM5r cut-over (live-play spacing metrics; 270-wide pred obs), 31434M steps. Cut-over export, **not** a final promote. |
| sha256 | `7933f34854f83f124641337471f7c66bd5d2f039bb6b686a013b0f5c3e462fd0` |

### tm4_30412M.bpk

| | |
|---|---|
| Lineage | TM4 latest at the TM5 cut-over (Nexto/Necto league; 270-wide pred obs), 30412M steps, then team_best as the cut checkpoint. Later superseded. |
| sha256 | `7591bd329dd7905c5a4d57611a63d8aebf9a6b7827ba0e55e1229e49c2c33691` |

### tm3_28999M.bpk

| | |
|---|---|
| Lineage | TM3 final candidate (ball prediction + nearest big pad, obs-only; 270-wide pred obs), 28999M steps. Then team_best; later superseded. |
| H2H | vs TM1 final: 2v2 **63.7%**, 3v3 **64.7%**; 1v1 vs ST25 **49.4%**; Galactus 1v1 **55.6%**. |
| sha256 | `316b29d0264a4b9f48d0a2bd09b79cd2fc0f12741ef3b0c9c44fd4d263000b1a` |

### tm2_27275M.bpk

| | |
|---|---|
| Lineage | TM2 final (team mechanics on top of TM1; team-ext 235-wide obs), 27275M steps. Then team_best; later superseded. Distinct from the mid-run `tm2_26089M.bpk` below. This is the RLBot pack's `weights/tm2` (same sha256, since 21f7fb2). |
| H2H | vs TM1 final: 2v2 **61.1%**, 3v3 **60.9%**; 1v1 vs ST25 **51.2%**; Galactus 1v1 **55.2%**. |
| sha256 | `3c1375fbe254869c7e69bfe67932c2a46311dd4d2fa913edbfbfb59d8b9d90d8` |

### tm2_26089M.bpk

| | |
|---|---|
| Lineage | TM2 (team mode: musty / flip-reset-use / double-tap / air double touch rewards, reset_shot and musty_setup setters, `tm2.toml`, binary `indococo-train-tm2-620e718`), continued from TM1 24275M, checkpoint `exp_TM2/checkpoints/1794756839` (26088.8M steps). Taken mid-run (TM2 still training); not the TM2 final. 235-input team-ext obs (`INDOCOCO_OBS_TEAM=1`), plays 1v1, 2v2 and 3v3. Was the pack's `weights/tm2` until 21f7fb2. |
| H2H (mid-run eval at this checkpoint) | 2v2 **58.0%** vs TM1 (116-84), 3v3 **56.5%** vs TM1 (113-87), 1v1 **49.0%** vs ST25 (98-102), 200 games each |
| Galactus (mid-run eval, not the 500-game panel) | 1v1 **62.5%** (125-75, 200 games); 2v2 9.6% (10-94), 3v3 4.8% (5-99) |
| Megabyte | 2v2 6.7% (7-97) |
| sha256 | `f76f1867a8d3090a8346ff59042995c59cf3efa9d641550aba621db0393c82e4` |

### tm1_24275M.bpk

| | |
|---|---|
| Lineage | TM1 final (first 2v2/3v3 team IPPO from ST36; team-ext 235-wide obs), 24275M steps. Then team_best; later superseded. |
| H2H | vs seed: 2v2 **85.7%**, 3v3 **88.5%**. |
| sha256 | `b5c752f90acae45b589d1cb71d2b088fc2c8a6ba4b18adbaa1bdab1d3ef61e5f` |

### st36_21349M.bpk

| | |
|---|---|
| Lineage | ST36 (PBRS `shadow_box` weight 30, gamma 0.998, not zero-sum, on the ST35 PFSP tree, `train_st36p.toml`, binary `indococo-train-st36-1cbe86c`), continued from ST35 20938.1M, selected chunk `1794658145` (21349.1M steps). 106-input obs (`INDOCOCO_OBS_EXT=1`). **Not promoted** (h2h 50.0 vs ST35, gate needs ≥52). ST25 remains current best. |
| H2H | **50.0%** vs ST35 (250-250) over 500 games |
| Pool | **92.88** on pool400 (ST35 93.0, ST25 93.33): 1v1.pt 81.2, 430M 84.5, Nexto 96.5, Necto 98.8, Element 99.5, Immortal 96.8 |
| Botpack5 | **69.99** avg (ST35 69.54, ST25 69.21): Ripple 71.0, Kira 71.4, Dice 64.2, Chronos 72.8, Galactus 40.8, Seer 99.8 |
| Galactus | **41.4%** over 500 games (ST35 37.6%, ST25 40.8%) |
| Megabyte | **84.2%** over 500 games (report-only; ST25 84.0%) |
| Gen panel | min **0.414** / mean **0.8149** (ST35 0.376 / 0.8126, ST25 0.408 / 0.8132) |
| sha256 | `7306467f253fb0213dbb9bb4c536ed91a2651ef1fbd16cb96b5b835fdcb18347` |

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

### st16_14786M.bpk

| | |
|---|---|
| Lineage | ST16, 14786.0M steps. **Promoted**, was 1v1 BEST until ST19r. 106-input obs. |
| H2H | **52.6%** vs ST15; FR4 74.6; pool400 92.04; bp5 65.83; Galactus 25.0%; Megabyte 71.6%. |
| sha256 | `e19c66598e88304a965d5cf0edc608f88ee62ca7f44c811fa6042839d6159b74` |

### st15_14024M.bpk

| | |
|---|---|
| Lineage | ST15, 14024.3M steps. **Promoted**, was 1v1 BEST until ST16. 106-input obs. |
| sha256 | `ba13d7cfa2cb12e767cce67cea6a351ba864642ba67614e718de50d7bd2cac5f` |

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

### st11_11090M.bpk

| | |
|---|---|
| Lineage | ST11, 11090M steps (ST8_final added to EXTRA_OPPS). **Promoted**, was 1v1 BEST until later ST runs. 106-input obs. |
| H2H | **54.8%** vs ST10; FR4 73.6; pool 92.6; bp5 65.8; Galactus 30.0%; Megabyte 70.4%. |
| sha256 | `687a9932dcd4b8b76277165afae249cab4f4832513ee160697cd049a0de93bbf` |

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
