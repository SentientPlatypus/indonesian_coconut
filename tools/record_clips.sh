#!/usr/bin/env bash
# Record clip JSONs from tools/clip_match.py to mp4 through RocketSimVis on a
# virtual display.   usage: tools/record_clips.sh <clip_dir> <out_dir>
set -euo pipefail
CLIPS=$1; OUT=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RSV_PY=${RSV_PY:-/home/ubuntu/rsv-venv/bin/python}
DISP=${DISP:-:99}
mkdir -p "$OUT"

Xvfb "$DISP" -screen 0 1440x960x24 -nolisten tcp >/dev/null 2>&1 &
XPID=$!
sleep 2
trap 'kill ${RPID:-} $XPID 2>/dev/null || true' EXIT
# llvmpipe start-up segfaults intermittently under heavy CPU load: retry
for try in 1 2 3 4 5; do
  (cd "$ROOT/RocketSimVis/src" && exec env DISPLAY=$DISP LIBGL_ALWAYS_SOFTWARE=1 \
    MESA_SHADER_CACHE_DISABLE=true QT_QPA_PLATFORM=xcb "$RSV_PY" main.py >/tmp/rsv.log 2>&1) &
  RPID=$!
  sleep 8
  kill -0 $RPID 2>/dev/null && break
  echo "RocketSimVis died on start (try $try), retrying" >&2
done
kill -0 $RPID 2>/dev/null || { echo "RocketSimVis failed to start" >&2; exit 1; }

for f in "$CLIPS"/*.json; do
  name=$(basename "$f" .json)
  # hold the first frame so interpolation/trails settle before recording
  python3 - "$f" hold <<'EOF'
import json, socket, sys, time
fr = json.load(open(sys.argv[1])); s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for _ in range(15):
    s.sendto(json.dumps(fr[0]).encode(), ("127.0.0.1", 9273)); time.sleep(1/15)
EOF
  dur=$(python3 -c "import json;print(len(json.load(open('$f')))/15+0.5)")
  ffmpeg -loglevel error -y -f x11grab -draw_mouse 0 -framerate 30 -video_size 1440x960 -i "$DISP" \
    -t "$dur" -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p "$OUT/$name.mp4" &
  FPID=$!
  python3 - "$f" <<'EOF'
import json, socket, sys, time
fr = json.load(open(sys.argv[1])); s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t0 = time.time()
for i, p in enumerate(fr):
    s.sendto(json.dumps(p).encode(), ("127.0.0.1", 9273))
    time.sleep(max(0.0, t0 + (i + 1) / 15 - time.time()))
EOF
  wait $FPID
  echo "recorded $OUT/$name.mp4"
done
