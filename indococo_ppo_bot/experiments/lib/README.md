# experiments/lib

Shared helpers for a future TM11 (or ST37) launcher. **Not wired into any
live script.** TM10 (`team_launch_tm10.sh`, `tm_TM10.env`, `tm10_window.sh`,
the waiter, `code_tm`, `ic_ppo`) must keep using its own copies until that
run finishes.

- `paths.env` — default `R`, LibTorch, `PYT`, eval-bot dirs, ST25 / TM9 / TM7 checkpoint paths.
- `chain.sh` — `run_eval` with `timeout --kill-after=60 ${EVAL_TIMEOUT:-5400}` and stderr in `$W/<name>.err`.
