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
| `v4_env.py`, `loop_config.py`, `rewards/`, `rl_math/` | Python reference reward, obs and config code. `indococo_ppo_bot/tools/parity.py` uses it to produce the expected rewards for the parity tests. |
| `eval_match.py` | Headless RocketSim evaluator used by the head-to-head evals against `1v1.pt`. |
| `rlbot_submission/` | The current Python championship submission. `policies/1v1.pt` is the warm-start and h2h reference. Left untouched. |
| `tools/` | RLBot server/GUI binaries and RLArenaCollisionDumper (for regenerating meshes). |
| `assets/` | README images. |

## Quick start

```sh
cd indococo_ppo_bot
cargo test --release                       # unit + 33 reward parity tests
cargo build --release --features torch,cuda --bin indococo-train
```

Training, warm start, critic warm-up, evals and deploy are covered in
[`indococo_ppo_bot/README.md`](indococo_ppo_bot/README.md).

The Python training stack (rlgym-ppo trainer, curriculum, A/B checkpoints, old
RLBot v4 bot) was removed from this branch. It is still on `master`.
