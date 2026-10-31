# checkpoints_to_test/

Policy weights worth testing, one `<name>.bpk` per checkpoint. New candidates get added here the same way.

## How to test one

1. Keep a single RLBot bot folder: either [`rlbot_indococo_ppo/`](../rlbot_indococo_ppo/) on this branch, or the folder from the [adb-832m release](https://github.com/SentientPlatypus/indonesian_coconut/releases/tag/adb-832m).
2. Copy the `.bpk` you want over `<bot>/weights/actor.bpk`, e.g. `cp checkpoints_to_test/adb_832M.bpk rlbot_indococo_ppo/weights/actor.bpk`.
3. Restart the bot (or the match). Weights load only at startup.

**Compatibility:** a `.bpk` only works with binaries built for the same obs, action and network layout. Every file here uses `V2DefaultObs`, the indococo action set and the current actor network. The `indococo-rlbot`/`.exe` binaries in `rlbot_indococo_ppo/` and in the release are byte-identical and run all of them. If the obs, actions or network change, rebuild the binaries and say so here.

## Checkpoints

Headless evals use seed 424242, the air-time clamp on (as deployed), a fresh engine per game and mirrored seeds. Pool = 400 games per opponent, win %.

### aud.pt
The AUD 1132M policy (same weights as `aud_1132M.bpk`) converted to the Python `DiscreteFF` state_dict format with `indococo_ppo_bot/tools/bpk_to_pt/`. It's byte-identical to `rlbot_submission/policies/1v1.pt` on master (sha256 `15e16300…53f68`). To test it in the Python bot: `cp checkpoints_to_test/aud.pt rlbot_submission/policies/1v1.pt`.

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
