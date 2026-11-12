import re, os
E="/home/ubuntu/ippo_runs/code_fr9/indococo_ppo_bot/experiments/st_ST30.env"; S="/opt/dlami/nvme/st31_stage/st_ST31.env"
t=open(E).read()
body=t[t.index("GAMMA="):]  # drop ST30's header comments, keep the code verbatim
body=body.replace("ST30_R","ST31_R").replace("${RUN:-ST30}","${RUN:-ST31}")
body=body.replace('# Start-point rule (as ST29): on a fresh start, use BEST\'s selected checkpoint.\n# The launcher\'s follow-up sets BESTF=st29_final when ST29 promotes (its\n# FINAL_CKPT is the checkpoint the launcher already passes), else the newest\n# promote=true final (st25_final -> exp_ST25/checkpoints/1794413355). START is\n# cleared so the launcher recomputes it (and END). No effect on resumes.',
 '# Start-point rule (as ST29/ST30): on a fresh start, use BEST\'s selected\n# checkpoint. The launcher\'s follow-up sets BESTF=st30_final when ST30 promotes\n# (its FINAL_CKPT is the checkpoint the launcher already passes), else the newest\n# promote=true final (st25_final -> exp_ST25/checkpoints/1794413355). START is\n# cleared so the launcher recomputes it (and END). No effect on resumes.')
body=body.replace('if [[ "${BESTF:-}" == */st29_final ]]; then TRAIN_TMPL=train_st30p.toml; else TRAIN_TMPL=train_st30.toml; fi',
                  'if [[ "${BESTF:-}" == */st30_final ]]; then TRAIN_TMPL=train_st31p.toml; else TRAIN_TMPL=train_st31.toml; fi')
code=[l for l in body.splitlines() if not l.lstrip().startswith("#")]; assert not [l for l in code if "st29" in l or "st30p" in l or "ST30" in l or "train_st30" in l], code
hdr="""# ST31 (passed the Nov 12 Discord/research/code cross-exam; reconciled design)
# = the start point's own recipe plus ONE change, in train_st31*.toml: 0.075 of
# the 0.175 fr7 normal-play replay (kept at 0.10) moves to a replay setter over
# real states 1.5-4.5 s before a NON-demo goal against ST25 by FR4, Galactus,
# Dice, Kira, Ripple and Chronos (st31_snap/preconcede_v2.f32). Capture seeds
# 5000-8549 (v1 windows 1.5/2.5/3.5 s) and 10000-13549 (v2 windows 1.5/2.25/
# 3.0/3.75/4.5 s), disjoint from diag 0-99, ST30's 1000-3399 and finals.
# Kickoff goals and goals within 3.5 s of a demo on us are excluded (ST30's
# lever), so ST31 is independent of ST30's outcome. Unsaveable states dropped
# (ball within 300 uu of our line, or projected in within 1 s with our car not
# goal-side / unable to reach); opponent mix by concede rate, floor 10%,
# Galactus 25% (resampled). Rewards unchanged (st.toml).
# Why: targeted setters are the only lever that moved Galactus; after demos,
# aerial and out-of-position goals are the biggest share of goals conceded to
# Galactus (ST29 22% + 12%) and to FR4 (32% + 22%). See EXPERIMENTS.md ST31.
# Base follows the start point (one change vs the start point's own recipe):
#  - ST30 promoted (BESTF=st30_final): train_st31p.toml = ST30's template
#    (ST25 + 0.05 demo_threat + 0.10 replay over gal_demo_v2.f32).
#  - otherwise (BESTF=st25_final): train_st31.toml = ST25's template.
# Same league as ST25 / ST30 (ST20r_final 2.0 via EXTRA_OPPS, FR4 0.816 in the
# toml, Galactus 6.429, Megabyte 2.0, Ripple 1.286 in the toml), OPPCH 0.70,
# 1 epoch, CHUNK 100M, reward st.toml (flip_reset 70), binary 9d4b2a6.
# Do NOT set RIPPLE_WEIGHT here: the trainer would override the toml.
"""
open(S+".tmp","w").write(hdr+body); os.replace(S+".tmp",S); print(open(S).read())
