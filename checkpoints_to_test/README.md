# checkpoints_to_test/

Policy weights worth testing, one `<name>.bpk` per checkpoint. New candidates get added here the same way.

## How to test one

1. Keep a single RLBot bot folder: either [`rlbot_indococo_ppo/`](../rlbot_indococo_ppo/) on this branch, or the folder from the [adb-832m release](https://github.com/SentientPlatypus/indonesian_coconut/releases/tag/adb-832m).
2. Copy the `.bpk` you want over `<bot>/weights/actor.bpk`, e.g. `cp checkpoints_to_test/adb_832M.bpk rlbot_indococo_ppo/weights/actor.bpk`.
3. Restart the bot (or the match). Weights load only at startup.

**Compatibility:** a `.bpk` only works with binaries built for the same obs, action and network layout. Every file here uses `V2DefaultObs`, the indococo action set and the current actor network. The `indococo-rlbot`/`.exe` binaries in `rlbot_indococo_ppo/` and in the release are byte-identical and run all of them. If the obs, actions or network change, rebuild the binaries and say so here.

## Checkpoints

Headless evals use seed 424242, the air-time clamp on (as deployed), a fresh engine per game and mirrored seeds. Pool = 400 games per opponent, win %.

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
