# checkpoints_to_test/

Each folder here is a complete RLBot v5 bot. To test one, copy the folder into your RLBot bot folder as is.

## rlbot_indococo_ppo_adb_832M (ADB 832M)

This is the ADB arm's best checkpoint (air-dribble bump + air_touch_height + victim-gated demo): `ckpt_831600000` (831.6M steps), warm-started from 430.8M. It contains `bot.toml`, `loadout.toml`, `weights/actor.bpk`, `indococo-rlbot.exe` (Windows) and `indococo-rlbot` (Linux). The binaries are the same as the 430M bot's; only the weights changed.

Headless results (2026-10-31; seed 424242, air-time clamp on, mirrored seeds):

| Match | W-L | Win % |
|---|---|---|
| ADB vs 430M | 636-364 | **63.6** |
| ADB vs X0-best | 521-479 | **52.1** |

Pool (400 games per opponent, win %):

| 1v1.pt | 430M | Nexto | Necto | Element | Immortal | Pool |
|---|---|---|---|---|---|---|
| 65.2 | 65.0 | 85.5 | 95.8 | 99.0 | 97.5 | **84.7** |

430M scored 79.0 on the same pool and 1v1.pt scored 77.5. Release: https://github.com/SentientPlatypus/indonesian_coconut/releases/tag/adb-832m
