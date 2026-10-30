# Indonesian Coconut: Rocket League PPO bot (Rust)

![Indonesian Coconut](assets/cover.png)

Indonesian Coconut is a 1v1 Rocket League bot trained with PPO on a discrete
90-action lookup table. On this branch it runs on
[indococo-ppo](https://github.com/SentientPlatypus/indococo-ppo), a Rust stack
built on RocketSim and Burn. Its reward set, observations and actions match
the E8B2 Python policy (`1v1.pt`), which it warm-starts from.

The Python policy beat Nexto 42–11 and Element 47–3 in RLBot matches
([replay](https://ballchasing.com/replay/dcdad7bc-640d-4eb7-bfd8-f0634e49cd30)).

## Repository layout

| Path | What it is |
|---|---|
| [`indococo_ppo_bot/`](indococo_ppo_bot/README.md) | The Rust bot: rewards (one module each, with parity tests against Python), obs/actions, the trainer (`train`, `import`, `export`, `record`) and the RLBot client. **Start here.** |
| `rlbot_indococo_ppo/` | RLBot v5 package: `bot.toml`, Linux and Windows binaries, `weights/actor.bpk`. |
| `collision_meshes/` | RocketSim soccar meshes. The trainer loads them from the repo root; the RLBot binary embeds them. |
| `assets/` | README images. |

## Quick start

```sh
cd indococo_ppo_bot
cargo test --release                       # unit + 33 reward parity tests
cargo build --release --features torch,cuda --bin indococo-train
```

Training, warm start, critic warm-up, evals and deploy are covered in
[`indococo_ppo_bot/README.md`](indococo_ppo_bot/README.md).

## What lives on `master`

This branch holds only the Rust bot. Everything else is still on `master`:

- the Python training stack (rlgym-ppo trainer, curriculum, A/B checkpoints, old RLBot v4 bot);
- the Python reference rewards and evaluator (`v4_env.py`, `loop_config.py`,
  `eval_match.py`, `rewards/`, `rl_math/`), needed only to regenerate the
  parity fixture or run h2h evals;
- `rlbot_submission/`, the live Python submission. Its `policies/1v1.pt` is
  the warm-start and h2h reference:
  `git show master:rlbot_submission/policies/1v1.pt > 1v1.pt`.

## External tools

- **RLBot v5 server and GUI**, to run `rlbot_indococo_ppo/` in matches:
  <https://rlbot.org> (download page) or
  <https://github.com/RLBot/core/releases> and
  <https://github.com/RLBot/gui/releases>.
- **RLArenaCollisionDumper**, only to regenerate `collision_meshes/` (for
  example after a Rocket League map update):
  <https://github.com/ZealanL/RLArenaCollisionDumper>. Run it with Rocket
  League open and copy the `.cmf` files into `collision_meshes/soccar/`.
