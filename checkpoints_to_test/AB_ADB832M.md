# ADB 832M (Rust indococo-ppo bot, ready-to-copy RLBot folder)

Copy **`checkpoints_to_test/rlbot_indococo_ppo_adb_832M/`** into your RLBot bot folder as is. It contains `bot.toml`, `loadout.toml`, `weights/actor.bpk`, `indococo-rlbot.exe` (Windows) and `indococo-rlbot` (Linux). The binaries are the same as the 430M bot's; only the weights changed.

## Headless (2026-10-31; seed 424242, air-time clamp on, mirrored seeds)

| Match | W-L | Win % |
|---|---|---|
| ADB vs 430M | 636-364 | **63.6** |
| ADB vs X0-best | 521-479 | **52.1** |

Pool (400 games per opponent, win %):

| 1v1.pt | 430M | Nexto | Necto | Element | Immortal | Pool |
|---|---|---|---|---|---|---|
| 65.2 | 65.0 | 85.5 | 95.8 | 99.0 | 97.5 | **84.7** |

430M scored 79.0 on the same pool and 1v1.pt scored 77.5.

## Lineage
- Branch `indococo-ppo`, ADB arm (air-dribble bump + air_touch_height + victim-gated demo)
- Checkpoint `ckpt_831600000` (831.6M steps), warm-started from 430.8M
- Release: https://github.com/SentientPlatypus/indonesian_coconut/releases/tag/adb-832m

## Watch in-game
- Strength vs Nexto, and how it feels compared with 430M
