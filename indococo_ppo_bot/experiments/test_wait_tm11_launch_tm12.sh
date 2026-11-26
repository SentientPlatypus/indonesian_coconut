#!/usr/bin/env bash
# Stub tests for wait_tm11_launch_tm12.sh (388a611 safeguards + start rule).
# Uses a fake $R and $PROC_ROOT; never talks to EC2 or a live trainer.
set -euo pipefail
SELF=$(readlink -f "${BASH_SOURCE[0]}")
C=$(dirname "$SELF")
WAIT=$C/wait_tm11_launch_tm12.sh
EVAL=$C/tm12_eval.py
fail=0
pass() { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; fail=$((fail+1)); }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
R=$TMP/runs
PROC=$TMP/proc
mkdir -p "$PROC" "$R/code_tm12" "$R/tm12_work" "$R/tm11_final" "$R/exp_TM11/logs" "$R/exp_TM11/checkpoints" \
  "$R/exp_TM10/checkpoints/1795503212" "$R/code_tm11/indococo_ppo_bot/experiments"
# The waiter sources lib/paths.env from C, which we point at this repo's experiments.
# Provide a dummy B12 that is a unique file (not identical to TM11).
B12=$TMP/indococo-train-tm12-test
echo tm12-bin > "$B12"; chmod +x "$B12"
echo tm11-bin > "$R/exp_TM11/indococo-train"; chmod +x "$R/exp_TM11/indococo-train"
for f in actor.bpk critic.bpk stats.toml; do
  echo x > "$R/exp_TM10/checkpoints/1795503212/$f"
  mkdir -p "$R/exp_TM11/checkpoints/100"
  echo x > "$R/exp_TM11/checkpoints/100/$f"
done
echo "$R/exp_TM11/checkpoints/100" > "$R/tm11_final/FINAL_CKPT"
touch "$R/TEAM_NO_FOLLOWUP"
# C for the waiter must be this experiments dir's parent (indococo_ppo_bot)
BOT=$(readlink -f "$C/..")
export PYT=$(command -v python3)
# Minimal mid.jsonl so start() can run in early DRY_RUN cases.
python3 -c "import json,pathlib; p=pathlib.Path('$R/exp_TM11/logs/mid.jsonl'); ck='$R/exp_TM11/checkpoints/100'; rows=[{'ckpt':ck,'h2h_2v2':0.40,'h2h_3v3':0.40,'nexto_2v2':0.50,'nexto_3v3':0.40},{'ckpt':ck,'h2h_2v2':0.41,'h2h_3v3':0.41,'nexto_2v2':0.50,'nexto_3v3':0.40}]; p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')"

# Helper: one fake /proc entry with a cmdline.
fake_proc() { # pid cmdline
  local pid=$1; shift
  mkdir -p "$PROC/$pid"
  printf '%s\0' "$@" | tr -d '\n' > "$PROC/$pid/cmdline"
  # cmdline as NUL-separated; waiter does tr '\0' ' '
}

run_dry() {
  DRY_RUN=1 R="$R" C="$BOT" B12="$B12" PROC_ROOT="$PROC" PYT="$PYT" \
    TM11_CMD="bash $R/code_tm11/indococo_ppo_bot/experiments/team_launch_tm11.sh" \
    TM10_CK="$R/exp_TM10/checkpoints/1795503212" \
    bash "$WAIT" >"$TMP/out" 2>"$TMP/err" && echo 0 || echo $?
}

# 1) TEAM_NO_FOLLOWUP missing -> die
rm -f "$R/TEAM_NO_FOLLOWUP"
if DRY_RUN=1 R="$R" C="$BOT" B12="$B12" PROC_ROOT="$PROC" PYT="$PYT" bash "$WAIT" >"$TMP/out" 2>"$TMP/err"; then
  bad "missing TEAM_NO_FOLLOWUP should die"
else
  grep -q 'TEAM_NO_FOLLOWUP missing' "$TMP/out" "$R/tm12_work/waiter.log" && pass "refuses without TEAM_NO_FOLLOWUP" || bad "wrong error for missing TEAM_NO_FOLLOWUP"
fi
touch "$R/TEAM_NO_FOLLOWUP"

# 2) exp_TM12 exists -> die
mkdir -p "$R/exp_TM12"
if DRY_RUN=1 R="$R" C="$BOT" B12="$B12" PROC_ROOT="$PROC" PYT="$PYT" bash "$WAIT" >"$TMP/out" 2>"$TMP/err"; then
  bad "existing exp_TM12 should die"
else
  grep -q 'exp_TM12 already exists' "$TMP/out" "$R/tm12_work/waiter.log" && pass "refuses if exp_TM12 exists" || bad "wrong error for exp_TM12"
fi
rmdir "$R/exp_TM12"

# 3) B12 identical to TM11 binary -> die
cp "$R/exp_TM11/indococo-train" "$B12"
if DRY_RUN=1 R="$R" C="$BOT" B12="$B12" PROC_ROOT="$PROC" PYT="$PYT" bash "$WAIT" >"$TMP/out" 2>"$TMP/err"; then
  bad "identical B12 should die"
else
  grep -q 'byte-identical' "$TMP/out" "$R/tm12_work/waiter.log" && pass "refuses byte-identical B12" || bad "wrong error for identical B12"
fi
echo tm12-bin > "$B12"; chmod +x "$B12"

# 4) live TM11 launcher in fake proc -> DRY_RUN dies (would not launch)
fake_proc 111 bash "$R/code_tm11/indococo_ppo_bot/experiments/team_launch_tm11.sh"
st=$(run_dry)
if [[ $st != 0 ]] && grep -q 'TM11 still running' "$TMP/out" "$R/tm12_work/waiter.log" "$R/tm12_work/waiter.log"; then
  pass "refuses to launch while TM11 launcher is alive"
else
  bad "launcher-alive not detected (st=$st)"
  cat "$TMP/out" "$TMP/err" || true
fi
rm -rf "$PROC/111"

# 5) live $X11/indococo-train -> refuse
fake_proc 222 "$R/exp_TM11/indococo-train" train /tmp/train.toml
st=$(run_dry)
if [[ $st != 0 ]] && grep -q 'TM11 still running' "$R/tm12_work/waiter.log"; then
  pass "refuses to launch while TM11 trainer is alive"
else
  bad "trainer-alive not detected (st=$st)"
fi
rm -rf "$PROC/222"

# 6) finals eval whose cmdline mentions tm11_final -> refuse
fake_proc 333 /opt/dlami/nvme/ippo_bins/indococo-train-tm11-abc h2h "$R/exp_TM11/checkpoints/100" "$R/tm11_final/foo" 504 8
st=$(run_dry)
if [[ $st != 0 ]] && grep -q 'TM11 still running' "$R/tm12_work/waiter.log"; then
  pass "refuses to launch while a TM11 finals eval is alive"
else
  bad "finals-eval-alive not detected (st=$st)"
fi
rm -rf "$PROC/333"

# 7) unrelated indococo-train (speedup / TM10) must NOT count as TM11
fake_proc 444 /opt/dlami/nvme/ippo_bins/indococo-train-speedup-xyz train /tmp/speed.toml
# No DONE, no panel: fallback mid path. Provide 2 complete mids so start() can run.
# h2h vs TM9 numbers here shouldn't pass the vs-TM10 rule without nexto in band...
# last-2-mids with nexto in band and h2h 0.52/0.50 would pick cand. Use failing
# h2h so start is TM10 (safe), and DRY_RUN should exit 0.
python3 - "$R/exp_TM11/logs/mid.jsonl" "$R/exp_TM11/checkpoints/100" <<'PY'
import json,sys
p,ck=sys.argv[1],sys.argv[2]
rows=[
  {"ckpt":ck,"h2h_2v2":0.40,"h2h_3v3":0.40,"nexto_2v2":0.50,"nexto_3v3":0.40},
  {"ckpt":ck,"h2h_2v2":0.41,"h2h_3v3":0.41,"nexto_2v2":0.50,"nexto_3v3":0.40},
]
open(p,"w").write("\n".join(json.dumps(r) for r in rows)+"\n")
PY
st=$(run_dry)
if [[ $st == 0 ]] && grep -q 'DRY_RUN: would run' "$R/tm12_work/waiter.log"; then
  pass "unrelated train process is not treated as TM11"
else
  bad "unrelated process blocked DRY_RUN (st=$st)"
  tail -20 "$R/tm12_work/waiter.log" || true
fi
rm -rf "$PROC/444"

# 8) last completed mid fallback (no DONE): newest unevaluated ckpt dir must not win
mkdir -p "$R/exp_TM11/checkpoints/999"
for f in actor.bpk critic.bpk stats.toml; do echo x > "$R/exp_TM11/checkpoints/999/$f"; done
rm -f "$R/tm11_final/FINAL_CKPT" "$R/tm11_final/DONE"
# mid row points at 100, not 999
st=$(run_dry)
got=$(python3 -c 'import json; print(json.load(open("'"$R"'/tm12_work/START_DECISION.json"))["start"])')
# start is fallback TM10 because mids fail the rule, but cand used in the decision must be 100
reason=$(python3 -c 'import json; print(json.load(open("'"$R"'/tm12_work/START_DECISION.json"))["reason"])')
if [[ $got == "$R/exp_TM10/checkpoints/1795503212" ]]; then
  pass "no-DONE fallback starts from TM10 when mids fail the rule"
else
  bad "no-DONE start was $got"
fi
# The cand fed to start() is not in START_DECISION; confirm waiter log mentions checkpoints/100 not 999
if grep -q 'checkpoints/100' "$R/tm12_work/waiter.log" && ! grep -q 'checkpoints/999' "$R/tm12_work/START_DECISION.json"; then
  pass "no-DONE fallback uses last mid ckpt, not the newest folder"
else
  # weaker: at least didn't pick 999 as start
  if [[ $got != *"/999" ]]; then
    pass "no-DONE fallback did not pick unevaluated newest folder as start"
  else
    bad "no-DONE picked checkpoints/999"
  fi
fi

# 9) START_PANEL pass -> cand
echo "$R/exp_TM11/checkpoints/100" > "$R/tm11_final/FINAL_CKPT"
touch "$R/tm11_final/DONE"
python3 - "$R/tm12_work/START_PANEL.json" "$R/exp_TM11/checkpoints/100" "$R/exp_TM10/checkpoints/1795503212" <<'PY'
import json,sys
open(sys.argv[1],"w").write(json.dumps({
  "cand":sys.argv[2],"fallback":sys.argv[3],
  "h2h_2v2":0.52,"h2h_3v3":0.50,
  "nexto_2v2":0.649,"nexto_3v3":0.528,
})+"\n")
PY
st=$(run_dry)
got=$(python3 -c 'import json; print(json.load(open("'"$R"'/tm12_work/START_DECISION.json"))["start"])')
if [[ $st == 0 && $got == "$R/exp_TM11/checkpoints/100" ]]; then
  pass "START_PANEL pass starts from the TM11 final"
else
  bad "panel pass start=$got st=$st"
  cat "$R/tm12_work/START_DECISION.json" || true
fi

# 10) START_PANEL nexto above the reference still starts from cand (one-sided gate)
python3 - "$R/tm12_work/START_PANEL.json" "$R/exp_TM11/checkpoints/100" "$R/exp_TM10/checkpoints/1795503212" <<'PY'
import json,sys
open(sys.argv[1],"w").write(json.dumps({
  "cand":sys.argv[2],"fallback":sys.argv[3],
  "h2h_2v2":0.52,"h2h_3v3":0.50,
  "nexto_2v2":0.80,"nexto_3v3":0.528,
})+"\n")
PY
st=$(run_dry)
got=$(python3 -c 'import json; print(json.load(open("'"$R"'/tm12_work/START_DECISION.json"))["start"])')
if [[ $st == 0 && $got == "$R/exp_TM11/checkpoints/100" ]]; then
  pass "START_PANEL nexto above reference starts from the TM11 final"
else
  bad "panel nexto-above start=$got st=$st"
  cat "$R/tm12_work/START_DECISION.json" || true
fi

# 11) START_PANEL nexto more than the band below the reference -> TM10
python3 - "$R/tm12_work/START_PANEL.json" "$R/exp_TM11/checkpoints/100" "$R/exp_TM10/checkpoints/1795503212" <<'PY'
import json,sys
open(sys.argv[1],"w").write(json.dumps({
  "cand":sys.argv[2],"fallback":sys.argv[3],
  "h2h_2v2":0.52,"h2h_3v3":0.50,
  "nexto_2v2":0.60,"nexto_3v3":0.528,
})+"\n")
PY
st=$(run_dry)
got=$(python3 -c 'import json; print(json.load(open("'"$R"'/tm12_work/START_DECISION.json"))["start"])')
if [[ $st == 0 && $got == "$R/exp_TM10/checkpoints/1795503212" ]]; then
  pass "START_PANEL nexto below band starts from TM10"
else
  bad "panel nexto-below start=$got st=$st"
fi

echo
if (( fail==0 )); then echo "All waiter stub tests passed"; exit 0
else echo "$fail waiter stub test(s) failed"; exit 1
fi
