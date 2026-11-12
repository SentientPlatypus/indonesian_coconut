"""Build ST31 templates (reconciled Nov 12 design) from the exact templates of the two possible start points."""
import os
E="/home/ubuntu/ippo_runs/code_fr9/indococo_ppo_bot/experiments/"; S="/opt/dlami/nvme/st31_stage/"
BANK="/home/ubuntu/ippo_runs/st31_snap/preconcede_v2.f32"
OLD='[[state_setters]]\nkind = "replay"\nweight = 0.175\npath = "@SNAP@"\n'
NEW=('[[state_setters]]\nkind = "replay"\nweight = 0.10\npath = "@SNAP@"\n'
     "# ST31's one change (reconciled after the Nov 12 cross-exam): 0.075 of the\n"
     "# 0.175 normal-play replay (fr7_snap, kept at 0.10) moves to a replay setter\n"
     "# over real states 1.5-4.5 s before a NON-demo goal against ST25 by FR4,\n"
     "# Galactus, Dice, Kira, Ripple and Chronos (st31_snap/preconcede_v2.f32;\n"
     "# kickoff goals and goals within 3.5 s of a demo on us excluded, that is\n"
     "# ST30's lever; unsaveable states filtered; opponent mix by concede rate,\n"
     "# floor 10%%, Galactus 25%%). Real-play replay mass stays 0.175. Start weight,\n"
     "# not step-weighted. Mirrored half the time (cars swapped), so either side is\n"
     "# the defender. Metric label \"replay\" (shared). Replay has no boost option.\n"
     '[[state_setters]]\nkind = "replay"\nweight = 0.075\npath = "%s"\n' % BANK)
HDR={"train_st31.toml":("train_st25.toml","ST31 = ST25's recipe (reward st.toml), used when ST30 does not promote"),
     "train_st31p.toml":("train_st30.toml","ST31p = ST30's recipe (ST25 + 0.05 demo_threat / 0.10 gal_demo_v2 replay), used when ST30 promotes")}
for out,(src,h) in HDR.items():
    t=open(E+src).read(); assert t.count(OLD)==1, src
    t=t.replace(OLD,NEW)
    t=("# %s,\n# with ONE change: 0.075 of the fr7 normal-play replay -> replay over\n# real pre-concede states (st31_snap/preconcede_v2.f32). Template copied\n# from %s; everything else byte-identical.\n" % (h,src))+t
    open(S+out+".tmp","w").write(t); os.replace(S+out+".tmp",S+out); print("wrote",out,"from",src)
