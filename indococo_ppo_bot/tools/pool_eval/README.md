# pool_eval: headless evals against an opponent pool

Plays a candidate policy against every bot in `pool.toml` in RocketSim
(RLGym v2 engine), first goal wins, and prints a table plus one JSON line per
opponent.

```bash
tools/pool_eval/fetch_bots.sh ~/eval_bots /path/to/1v1.pt /path/to/430M/policy.safetensors
python tools/pool_eval/pool_eval.py --candidate checkpoints/<n> --games 400 --procs 32 \
    --out pool.jsonl --label <n>
python tools/pool_eval/test_poolkit.py          # quick harness checks, no weights needed
```

`--candidate` takes a Rust checkpoint directory (exported on the fly with
`indococo-train export`; set `TRAIN_BIN`), an exported `.safetensors`, or a
DiscreteFF `.pt` state dict. It needs Python with `rlgym` 2.x, `rocketsim`
and `torch`. `--only nexto,necto` limits the pool. The pool score is the mean
decisive win rate over the opponents.

## Protocol

- 1v1 Octane, RLGym `KickoffMutator` spawn from the seeded numpy RNG. A game
  ends at the first goal, or as a timeout after 30 s without a touch or 300 s.
- Each seed is played twice, candidate blue then orange. Every game uses a
  fresh engine, so the same seed replays the same game. Each car's spawn gets
  a seeded jitter (up to 60 uu, 4 degrees; ball stays centered): with only 5
  kickoff spawns, two deterministic bots would otherwise play just 10
  distinct games.
- The engine advances to the next tick any agent has to decide on: every
  8 ticks for our policy, 6 for Immortal, every tick during a scripted kickoff.
  Actions apply on the tick they are chosen (RLBot runs them a few ticks late).
- Our policy plays greedily (argmax) on DefaultObs with the 1.25 s air-time
  clamp, which is what the deployed bot sees. `--no-air-clamp` (or
  `clamp_air_time = false` on a pool entry) drops it to match master's
  `h2h_1v1.py`, which built its env without the clamp.
- Public bots run their own unmodified obs builders, models and kickoff code
  through `poolkit/v1compat.py`: an RLGym v1 `GameState`/`PlayerData` view of
  the v2 state (rlgym_compat's conversion) and stub `rlgym_compat`/`rlbot`
  modules. Nexto keeps its stochastic sampling (beta 1, 0.5 at kickoffs);
  Necto, Element and Immortal play as in their `bot.py`.

## Pool

| bot | source | license | weights | obs / actions | tick skip | headless |
|---|---|---|---|---|---|---|
| 1v1.pt | master `rlbot_submission/policies` | ours | DiscreteFF state dict | DefaultObs / 90-row table | 8 | exact |
| ppo_430M | `ippo_runs/e8b2_1v1/bots/ckpt_430M` | ours | safetensors | same | 8 | exact |
| Nexto | github.com/VirxEC/NectoFamily `nexto/` (Rolv-Arild) | none stated | TorchScript `nexto-model.pt` | NextoObsBuilder (attention over all entities) / 90-row table | 8 | yes; speedflip kickoff script |
| Necto | same repo, `necto/` | none stated | TorchScript `necto-model.pt` | NectoObsBuilder / factored 5-way | 8 | yes; same kickoff script |
| Element | github.com/VirxEC/Elementv5 | none stated | pickled torch `model.p` | CustomObs (2 cars) / 5 categorical + 3 Bernoulli | 8 | yes; its Speedflip sequence |
| Immortal | github.com/RLBot/RLBotPack `RLBotPack/Immortal` | MIT (pack) | TorchScript `jit.pt` | ExpandAdvancedObs / 90-row table | 6 | yes |
| zero / random | `poolkit/agent.py` | | | | 8 | sanity baselines |

Not in the pool: Seer (RLBotPack, LSTM fed raw RLBot packet fields such as
euler angles and per-wheel contact; would need its own packet shim), Ripple
and Tecko (no public weights found). `fetch_bots.sh` pins the commits; the
weights stay in `~/eval_bots`, not in this repo, because the Necto family and
Element repos state no license.

Adapter checks: every public bot beats `zero` and `random` in all test games,
1v1.pt against itself is 18-22 over 40 jittered games, and the public bots
rank sensibly among themselves (Necto beats Element 33-7, Immortal beats
Element 31-9).

## Keep-best and the eval loop

`keep_best.py` pool-evaluates one checkpoint and copies it (checkpoint dir,
`policy.safetensors`, `best.json`) to a best directory when its pool score
beats the current best's and it also wins a head-to-head against the current
best. `tools/eval_loop.sh` runs it every `EVERY` steps during training:

```bash
TRAIN_BIN=indococo-train PY=python tools/eval_loop.sh <checkpoints> logs/evals.jsonl bots/best
```
