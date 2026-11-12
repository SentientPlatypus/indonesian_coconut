# ST31 tools (pre-concede replay bank)
These scripts run on the EC2 box from `/opt/dlami/nvme/st31_stage`. Paths are hard-coded.
- `capture_preconcede.py` (v1 windows 1.5/2.5/3.5 s) and `capture_preconcede_v2.py` (1.5/2.25/3.0/3.75/4.5 s) record states before non-demo, non-kickoff goals against CAND by one poolkit opponent.
- `capture.sh`, `capture2.sh` and `capture3.sh` are the capture batches for seeds 5000-8549 and 10000-13549.
- `build_bank_v2.py` applies the unsaveable filter, resamples the opponent mix and validates the bank. It writes `preconcede_v2.f32` and `preconcede_v2.json`.
- `mk_tmpl.py` and `mk_env.py` build `train_st31*.toml` and `st_ST31.env` from ST25's and ST30's files.
- `render.sh`, `check.py` and `envtest.sh` render and check both branches.
- `measure.py` and `measure.sh` measure the step share with `rates`.
