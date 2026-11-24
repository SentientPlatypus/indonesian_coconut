#!/usr/bin/env bash
# Report-only 504-game panel: TM11 final vs TM10 final (both modes), plus Nexto
# taken from TM11 finals if complete, else the last mid. Never edits the TM11
# launcher, team_best, or BEST. Writes $OUT/h2h_vs_tm10_{2v2,3v3}.json and
# $OUT/START_PANEL.json for tm12_eval.py start --report.
#   B12=<bin> bash tm11_vs_tm10_report.sh
set -u
R=${R:-/home/ubuntu/ippo_runs}
C=${C:-$R/code_tm12/indococo_ppo_bot}
F11=$R/tm11_final
X11=$R/exp_TM11
OUT=${OUT:-$R/tm12_work}
TM10=${TM10_CK:-$R/exp_TM10/checkpoints/1795503212}
GAMES=${GAMES:-504}
THREADS=${THREADS:-8}
TOUT=${TOUT:-10800}
LOCK=$OUT/tm11_vs_tm10.lock
source "$C/experiments/lib/paths.env"
mkdir -p "$OUT"
log() { echo "$(date '+%F %T %Z') $*" | tee -a "$OUT/report.log"; }

# Prefer the TM12 binary; fall back to the already-finished TM11 copy (read-only).
B=${B12:-${B:-}}
if [[ -z ${B:-} ]]; then
  if [[ -x ${X11}/indococo-train ]]; then B=$X11/indococo-train
  else log "BLOCKED: set B12 to the TM12 binary (or wait for TM11 to finish)"; exit 1
  fi
fi
[[ -x $B ]] || { log "BLOCKED: binary $B missing or not executable"; exit 1; }
[[ -f $F11/DONE ]] || { log "BLOCKED: $F11/DONE missing; wait for TM11 finals"; exit 1; }
ck=${CK:-$(cat "$F11/FINAL_CKPT")}
[[ -s $ck/actor.bpk ]] || { log "BLOCKED: TM11 final $ck has no actor.bpk"; exit 1; }
[[ -s $TM10/actor.bpk ]] || { log "BLOCKED: TM10 checkpoint $TM10 has no actor.bpk"; exit 1; }

count_heavy_jobs() {
  local found_train=0 found_fid=0 found_eval=0 cmd
  for dir in ${PROC_ROOT:-/proc}/[0-9]*; do
    cmd=$(tr '\0' ' ' <"$dir/cmdline" 2>/dev/null) || continue
    case $cmd in
      *gal_fidelity*) found_fid=1 ;;
      *indococo-train\ train*) found_train=1 ;;
      *indococo-train*) found_eval=1 ;;
    esac
  done
  echo $((found_train + found_fid + found_eval))
}

if [[ ${FORCE:-0} != 1 && -s $OUT/START_PANEL.json && -s $OUT/h2h_vs_tm10_2v2.json && -s $OUT/h2h_vs_tm10_3v3.json ]]; then
  log "already have $OUT/START_PANEL.json (set FORCE=1 to overwrite)"; exit 0
fi

mkdir -p "$OUT"
exec 9>"$LOCK"
if ! flock -n 9; then
  log "BLOCKED: another tm11_vs_tm10_report.sh holds $LOCK"; exit 1
fi

heavy=$(count_heavy_jobs)
if [[ ${HEAVY_OK:-0} != 1 && $heavy -ge 2 ]]; then
  log "BLOCKED: $heavy heavy jobs already running (≤2 rule); set HEAVY_OK=1 to override"; exit 1
fi

export LD_LIBRARY_PATH=${LIBTORCH_LIB:-${LD_LIBRARY_PATH:-}}
export INDOCOCO_OBS_EXT=1 INDOCOCO_OBS_TEAM=1 INDOCOCO_OBS_PRED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
cd "$MESH_REPO"

for n in 2 3; do
  out=$OUT/h2h_vs_tm10_${n}v${n}.json
  err=$OUT/h2h_vs_tm10_${n}v${n}.err
  log "h2h ${n}v${n}: $B h2h $ck $TM10 $GAMES $THREADS (timeout ${TOUT}s)"
  set +e
  timeout --kill-after=60 "$TOUT" env INDOCOCO_TEAM_SIZE=$n nice -n 19 "$B" h2h "$ck" "$TM10" "$GAMES" "$THREADS" \
    >"$out.tmp" 2>"$err"
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    rm -f "$out.tmp"
    log "FAILED: ${n}v${n} exit $status (stderr $err)"; exit 1
  fi
  tail -1 "$out.tmp" >"$out"
  rm -f "$out.tmp"
  log "wrote $out"
done

# Nexto: TM11 finals 504-g if complete, else last mid (200 g, noisier).
EVP="$PYT $C/experiments/tm12_eval.py"
"$PYT" - "$F11" "$X11/logs/mid.jsonl" "$OUT" "$ck" "$TM10" <<'PY'
import json, math, os, sys

def last_json(path):
    try:
        with open(path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        d = json.loads(lines[-1]) if lines else {}
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}

def num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None

def wr(d, expect=0):
    g, w, l = num(d.get("games")), num(d.get("wins")), num(d.get("losses"))
    if g is None or w is None or l is None or w + l <= 0 or g < 0.9 * expect:
        return None
    return num(d.get("win_rate_decisive"))

F11, mid_path, out, ck, tm10 = sys.argv[1:6]
nexto, nexto_src = {}, {}
for n in (2, 3):
    d = last_json(os.path.join(F11, f"nexto_{n}v{n}.json"))
    v = wr(d, 504)
    src = "tm11_final" if v is not None else None
    if v is None:
        # last complete mid
        v = None
        try:
            rows = [json.loads(l) for l in open(mid_path) if l.strip()]
        except OSError:
            rows = []
        for row in reversed(rows):
            v = num(row.get(f"nexto_{n}v{n}"))
            if v is not None:
                src = "last mid"
                break
    nexto[n] = v
    nexto_src[n] = src

h, team = {}, {}
for n in (2, 3):
    d = last_json(os.path.join(out, f"h2h_vs_tm10_{n}v{n}.json"))
    h[n] = wr(d, 450)
    team[n] = (d.get("team") or {})

panel = {
    "cand": ck,
    "fallback": tm10,
    "h2h_2v2": h[2],
    "h2h_3v3": h[3],
    "nexto_2v2": nexto[2],
    "nexto_3v3": nexto[3],
    "nexto_src_2v2": nexto_src[2],
    "nexto_src_3v3": nexto_src[3],
    "team_a": {f"{k}_{n}v{n}": (team[n].get("a") or {}).get(field)
               for n in (2, 3)
               for k, field in (("touch", "touches_per_car_min"),
                                ("depth", "deepest_car_depth_uu"),
                                ("ncr", "no_challenger_ratio"))},
    "team_b": {f"{k}_{n}v{n}": (team[n].get("b") or {}).get(field)
               for n in (2, 3)
               for k, field in (("touch", "touches_per_car_min"),
                                ("depth", "deepest_car_depth_uu"),
                                ("ncr", "no_challenger_ratio"))},
}
path = os.path.join(out, "START_PANEL.json")
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(panel, f)
    f.write("\n")
os.replace(tmp, path)
print(json.dumps(panel))
PY
log "wrote $OUT/START_PANEL.json"
log "done: TM11 vs TM10 report in $OUT"
