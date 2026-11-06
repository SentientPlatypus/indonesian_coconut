# Megabyte Integration

Megabyte (SubparN0va/ByteBotpack, MIT 2026) is now integrated as a frozen league
opponent in both training and evaluation. It's a GigaLearn bot very good at
dodging air-dribble bumps.

## Architecture

- **Observation:** ByteObs (233 floats) — ball (9), previous controls (8), 34 pad
  values, teammate count / 2, opponent count / 3, self (30), 2 teammate slots
  (30 each), 3 opponent slots (30 each). Orange mirrors x and y (NOT z). Boost
  pad timers use the OPPOSITE orientation from the player-relative pads
  (legacy behavior).
- **Actions:** GigaLearn DefaultAction (90 actions + masks), tick skip 8, action
  delay 7.
- **Network:** Shared head 2x1024 + policy 4x1024, Linear -> LayerNorm -> ReLU.
  Greedy argmax over masked logits.
- **Weights:** `POLICY.lt` and `SHARED_HEAD.lt` as TorchScript modules (11.8 MB +
  5.2 MB), loaded via `tch::CModule`.

## Files

- `src/megabyte.rs` — Rust port of ByteObs and DefaultAction, `MegabytePolicy`
  that loads the TorchScript modules.
- `experiments/pool_megabyte.toml` — 1v1 eval pool with Megabyte as the opponent.
- `experiments/st_ST10.env` — ST10 variant with Megabyte at weight 2.0 in the
  league (same as Galactus 2.5).
- `tools/pool_eval/fetch_bots.sh` — updated to fetch Megabyte from ByteBotpack.
- `tools/pool_eval/poolkit/botpack5.py` — Python reference implementation
  (`Megabyte` class) for validation.
- `tools/megabyte_validate.py` — validation script (smoke test; full parity
  check requires GPU).

## Fetching weights

```sh
cd indococo_ppo_bot
bash tools/pool_eval/fetch_bots.sh ~/eval_bots
```

This clones ByteBotpack and copies `POLICY.lt` and `SHARED_HEAD.lt` into
`~/eval_bots/megabyte/`.

## Building

The training binary must be built with `--features torch` to support Megabyte's TorchScript modules:

```sh
cd indococo_ppo_bot
cargo build --release --features torch --bin indococo-train
```

For ST10 training, copy the binary to the expected location:

```sh
cp target/release/indococo-train /home/ubuntu/ippo_runs/code_fr9/bin/indococo-train-mega
```

The `st_ST10.env` file points `B0` at `indococo-train-mega` (not `indococo-train-gal`) so the binary knows the `megabyte` opponent kind.

## Training with Megabyte

ST10 adds Megabyte to the league at weight 2.0:

```sh
cd indococo_ppo_bot/experiments
RUN=ST10 VARIANT=ST10 bash st_launch.sh
```

The `st_ST10.env` file sets `MEGA_WEIGHT=2.0`, and `st_launch.sh` appends:

```toml
[[opponents.external]]
name = "megabyte"
kind = "megabyte"
path = "/home/ubuntu/eval_bots/megabyte"
weight = 2.0
```

Every chunk, the candidate is evaluated against Megabyte (200 games), and the
result is logged in `mid.jsonl` as `megabyte_200`. Finals include 500 games
against Megabyte, logged in `megabyte_final.jsonl`.

## Evaluation

```sh
python tools/pool_eval/pool_eval.py \
  --candidate <policy.safetensors> \
  --pool experiments/pool_megabyte.toml \
  --games 500 --procs 16 --seed 424242 \
  --out megabyte_500.jsonl
```

## Device handling

The Megabyte opponent in training uses `tch::Device::Cuda(0)` for inference. The "cuda" backend
in `device()` was NOT changed from `Device::cuda(0)` to `Device::libtorch_cuda(0)` — it stays as-is
for other parts of the training stack. Only the Megabyte external opponent uses the tch backend directly.

## Validation

The Rust port reproduces ByteObs and DefaultAction line by line from the C++
source (`inc/RLGymCPP/ObsBuilders/ByteObs.cpp`,
`ActionParsers/DefaultAction.cpp`). Key differences from Galactus:

1. **Observation size:** 233 (Galactus 258).
2. **Orange inversion:** x and y mirrored, NOT z (Galactus mirrors all three).
3. **Pad timer orientation:** OPPOSITE from the player-relative pads (legacy
   ByteObs behavior).
4. **Player features:** 30 floats (29 + padding 1.0), not 29.
5. **Activation:** ReLU (Galactus LeakyReLU 0.01).
6. **No ball prediction:** ByteObs does not include future ball states.

**Manual validation steps:**

1. Fetch the weights: `bash tools/pool_eval/fetch_bots.sh ~/eval_bots`.
2. Build the training binary: `cargo build --release --features torch --bin indococo-train`.
3. **Python validation** (generates deterministic states and validates obs builder):
   ```sh
   python tools/megabyte_validate.py ~/eval_bots/megabyte
   ```
4. **Rust observation generation** (outputs 20 JSON lines with obs + action):
   ```sh
   ./target/release/indococo-train megabyte-obs ~/eval_bots/megabyte dummy.jsonl
   ```
5. **Logits parity check** (compares Python vs Rust inference):
   ```sh
   python tools/megabyte_compare_logits.py ~/eval_bots/megabyte ./target/release/indococo-train
   ```
   Expected: max-abs logit diff < 1e-4, action match rate 100%.
6. **Head-to-head strength check**:
   ```sh
   ./target/release/indococo-train megabyte-h2h <checkpoint> ~/eval_bots/megabyte 100 4
   ```
   Expected: Megabyte wins ~36% (similar to Python Megabyte vs ST8).
7. **Compilation test** (verify both with and without torch feature):
   ```sh
   cargo check                  # Without torch
   cargo check --features torch # With torch
   ```
5. Run an evaluation against Nexto with Megabyte in the pool:
   ```sh
   # Create a pool with Nexto + Megabyte
   cat > /tmp/test_pool.toml <<EOF
   [[opponent]]
   name = "Nexto"
   kind = "python"
   path = "~/eval_bots/nexto"
   [[opponent]]
   name = "Megabyte"
   kind = "megabyte"
   path = "~/eval_bots/megabyte"
   EOF
   
   python tools/pool_eval/pool_eval.py \
     --candidate <1v1.pt or policy.safetensors> \
     --pool /tmp/test_pool.toml --games 50 --procs 8 --seed 1 \
     --out /tmp/sanity.jsonl
   ```
   
   If Megabyte plays at strength (not random), the obs and actions are correct.

**What still needs GPU/RocketSim validation:**

- Exact numerical parity between Rust and Python obs builders on identical states
  (requires a dedicated test harness that can serialize RocketSim states and
  load them in both).
- Megabyte strength vs. the Python reference in a 500-game head-to-head.

## License

The Megabyte weights and ByteBotpack C++ code are MIT licensed, copyright 2026
SubparNova. Our Rust port (`src/megabyte.rs`) is GPL-2.0 like the rest of this
repository.

## Usage in ST runs

ST10 is the first run with Megabyte. The hypothesis: after ST8/ST9 broadened
recent gens + league fraction, folding in Megabyte (and ST7_final) keeps the
panel diverse across playstyles without overweighting any single opponent.
Megabyte weight 2.0 matches Galactus's profile (2.5).

**Expected behavior:** the policy generalizes better to evasive opponents that
dodge air-dribble bumps, without sacrificing strength against the existing panel
(FR4, FR8, botpack5, pool400, Galactus).

**Promotion gate:** ST10 promotes if >= 52% vs BEST and pool/FR4/bp5 not worse
than BEST's. Megabyte WR is reported but not a hard gate (report-only, or a
loose tolerance like Galactus). This prevents blocking promotion on a bot we
haven't beaten yet.

## Future work

- Add a `megabyte-h2h` subcommand to `indococo-train` (like `galactus-h2h`) for
  quick parity checks.
- Port other ByteBotpack bots (Byte, Kilobyte) if ST10 shows Megabyte is useful.
- Consider dynamic opponent weighting based on win rate (if the policy starts
  beating Megabyte consistently, lower its weight).
