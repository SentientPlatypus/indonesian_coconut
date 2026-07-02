# checkpoints_to_test/

Non-gitignored drop zone for the policies worth testing locally. The EC2 loop
(see `LOOP.md` §3f) refreshes these with `tools/export_best.py` after every
PROMOTE and commits + pushes them; on your local machine just:

```bash
git pull
# then point RLBot / Rlgym-v2-to-rlbot-v5 at:
#   checkpoints_to_test/PPO_POLICY_V4_BEST.pt     (all-time best by eval score)
#   checkpoints_to_test/PPO_POLICY_V4_LATEST.pt   (tip of the current training line)
# and read STATUS.md for scores, style metrics, and the config that produced them.
```

Filenames are stable and overwritten on each export so the folder stays two
~30MB files (git history still grows per export — that's the accepted cost of
pull-and-test convenience).
