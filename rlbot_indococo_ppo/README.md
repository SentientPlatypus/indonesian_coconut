# rlbot_indococo_ppo

An RLBot v5 package for the indococo-ppo 1v1 bot. Source and training notes:
`../indococo_ppo_bot/README.md`.

| File | Purpose |
|---|---|
| `bot.toml` | Agent config. Uses `run_command` on Windows and `run_command_linux` on Linux. |
| `indococo-rlbot.exe` | Windows x86_64 binary (cross-built with `x86_64-pc-windows-gnu`). |
| `indococo-rlbot` | Linux x86_64 binary. |
| `weights/actor.bpk` | Policy weights. The current file is `1v1.pt` imported into Burn. |
| `loadout.toml` | Copy of master's `rlbot_submission/loadout.toml`. |

Both binaries run inference on the CPU with Burn and embed the collision
meshes, so the package needs neither Python nor a GPU. To deploy a newly
trained policy, copy `actor.bpk` from a training checkpoint into `weights/`.

To rebuild the binaries, run these from `indococo_ppo_bot/`:

    cargo build --profile dist --features rlbot --bin indococo-rlbot
    CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc \
      cargo build --profile dist --features rlbot --bin indococo-rlbot --target x86_64-pc-windows-gnu

FR8 and later checkpoints take a 106-float observation (base obs + previous
action + car-frame ball terms). The binaries detect this from the input width of
`weights/actor.bpk` at startup, so the same binaries run FR7-and-older (92) and
FR8 (106) weights: just copy the checkpoint's `actor.bpk` into `weights/`.
