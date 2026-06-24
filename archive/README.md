# Archive

Older scripts kept for reference. **Not used by the current pipeline** and not
guaranteed to run against the current dependencies.

- `rlgymexample.py` — earlier rlgym v2 trainer that preceded `freestyler.py`.
- `example.py` — the original trainer, built on the older **`rlgym_sim`** framework
  (different API from the current rlgym v2). Depends on the two helpers below.
- `customActionParser.py` — `AdvancedLookupTableAction`, used only by `example.py`.
- `rocketsimvis_rlgym_sim_client.py` — RocketSimVis client for the `rlgym_sim` path,
  used only by `example.py`. (The current trainers use `../rsv_renderer.py` instead.)

The active training entry point is `../freestyler.py`; start new runs from
`../train_template.py`.
