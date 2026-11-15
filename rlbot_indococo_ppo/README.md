# rlbot_indococo_ppo

An RLBot v5 package for the indococo-ppo bots (1v1, 2v2, 3v3). Source and
training notes: `../indococo_ppo_bot/README.md`.

| File | Purpose |
|---|---|
| `bot.toml` | **Indonesian Coconut**: ST25 in 1v1, TM2 in 2v2/3v3 (the default). |
| `tm2.bot.toml` | **Indonesian Coconut TM2**: TM2 in every mode, 1v1 included. |
| `indococo-rlbot.exe` | Windows x86_64 binary (cross-built with `x86_64-pc-windows-gnu`). |
| `indococo-rlbot` | Linux x86_64 binary. |
| `weights/st25/actor.bpk` | ST25 1v1 policy, 20728M steps (= `checkpoints_to_test/st25_20728M.bpk`, 106-wide 1v1 obs). |
| `weights/tm2/actor.bpk` | TM2 team policy, 26089M steps (= `checkpoints_to_test/tm2_26089M.bpk`, 235-wide team-ext obs). |
| `weights/actor.bpk` | The original `1v1.pt` import. Kept for reference; not used unless a flag points at it. |
| `loadout.toml` | Indonesian (red/white) car: Fennec, Stripes decal, Indonesia flag antenna, Palm Tree topper; item ids from RLBotGUI items.csv. |

Both binaries run inference on the CPU with Burn and embed the collision
meshes, so the package needs neither Python nor a GPU.

## Choosing the model (ST25 or TM2 in 1v1)

When the bot starts it reads the team sizes of the match and picks:

* **1v1**: the `--1v1` model. Default `st25`.
* **2v2 / 3v3** (any team with more than one car): the `--team` model. Default `tm2`.

Easiest way, in the RLBot GUI: add this folder, then drag in

* **Indonesian Coconut** (`bot.toml`) for ST25 in 1v1, or
* **Indonesian Coconut TM2** (`tm2.bot.toml`) for TM2 in 1v1.

Both play TM2 in 2v2 and 3v3, and you can mix them in one match.

The flags behind them (in `run_command` / `run_command_linux`):

    indococo-rlbot.exe --1v1 st25 --team tm2    # bot.toml (default)
    indococo-rlbot.exe --1v1 tm2  --team tm2    # tm2.bot.toml

| Flag | Env var | Default | Values |
|---|---|---|---|
| `--1v1 <model>` | `INDOCOCO_1V1_MODEL` | `st25` | `st25`, `tm2`, or a checkpoint directory |
| `--team <model>` | `INDOCOCO_TEAM_MODEL` | `tm2` | `tm2`, or a checkpoint directory |
| | `INDOCOCO_WEIGHTS` | | a checkpoint directory; overrides both |

A model name is a folder under `weights/` holding `actor.bpk`, so a new
policy is just `weights/<name>/actor.bpk` plus `--1v1 <name>` or
`--team <name>`. A value with a `/` or `\` is a checkpoint directory path.
Flags beat env vars. A 1v1-only net (ST runs) cannot play 2v2/3v3; the bot
refuses to start with one as `--team`.

The obs follows the chosen net's input width: 92 (FR7 and older), 106
(FR8 and ST runs), 235 (TM1/TM2 team-ext, any team size, 1v1 included) and
270 (TM3 team-ext + ball prediction). TM2 in 1v1 therefore uses the team-ext
obs with empty ally slots, the same obs it was evaluated with in training.

## Checking it offline

    indococo-rlbot.exe --selftest 1               # ST25 in 1v1
    indococo-rlbot.exe --selftest 1 --1v1 tm2     # TM2 in 1v1
    indococo-rlbot.exe --selftest 2               # TM2 in 2v2
    indococo-rlbot.exe --selftest 3               # TM2 in 3v3

Each runs a RocketSim match with every car driven by the chosen policy and
prints the model, obs width and an action summary ending in `OK`. No Rocket
League or RLBot needed. In a live match, the startup line in the RLBot
console (`indococo-rlbot: 1v1 -> ...weights\st25 (actor input 106, ...)`)
shows which model was picked.

## Rebuilding the binaries

The committed binaries are current, so you only need this after changing
`src/bin/rlbot.rs`. From `indococo_ppo_bot/` on Linux:

    cargo build --profile dist --features rlbot --bin indococo-rlbot
    CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc \
      cargo build --profile dist --features rlbot --bin indococo-rlbot --target x86_64-pc-windows-gnu

then copy `target/dist/indococo-rlbot` and
`target/x86_64-pc-windows-gnu/dist/indococo-rlbot.exe` into this folder.
