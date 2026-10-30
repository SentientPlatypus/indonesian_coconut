#!/usr/bin/env bash
# Downloads the pool bots into ~/eval_bots/<name>/ at pinned commits.
# Weights are not committed to this repo (licenses unclear); this script is the source of truth.
#
#   tools/pool_eval/fetch_bots.sh [dest=~/eval_bots] [path/to/1v1.pt] [path/to/430M policy.safetensors]
set -euo pipefail
DEST=${1:-$HOME/eval_bots}
SRC=$DEST/src
mkdir -p "$SRC"

clone() {  # url dir commit
  [[ -d $SRC/$2 ]] || git clone -q "$1" "$SRC/$2"
  git -C "$SRC/$2" fetch -q --depth 1 origin "$3" 2>/dev/null || true
  git -C "$SRC/$2" checkout -q "$3"
}
clone https://github.com/VirxEC/NectoFamily NectoFamily e5510350299e15a850d4388705ad4ee7dfe0e9ed
clone https://github.com/VirxEC/Elementv5 Elementv5 fd33e506ffe6a78277eb098a245a1c08ed1f33ed
if [[ ! -d $SRC/rlbotpack_v4 ]]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/RLBot/RLBotPack "$SRC/rlbotpack_v4"
  git -C "$SRC/rlbotpack_v4" sparse-checkout set RLBotPack/Immortal RLBotPack/Seer
fi
git -C "$SRC/rlbotpack_v4" checkout -q 0033a98d4e060670f334f60f7f8dc5ee21ba93bb

mkdir -p "$DEST"/{nexto,necto,element,immortal}
cp "$SRC"/NectoFamily/nexto/{agent.py,nexto_obs.py,nexto-model.pt} "$DEST/nexto/"
cp "$SRC"/NectoFamily/necto/{agent.py,necto_obs.py,necto-model.pt} "$DEST/necto/"
cp -r "$SRC"/Elementv5/src/{agent.py,obs.py,model.p,sequences} "$DEST/element/"
cp -r "$SRC"/rlbotpack_v4/RLBotPack/Immortal/{agent.py,jit.pt,obs,action} "$DEST/immortal/"

if [[ -n ${2:-} ]]; then mkdir -p "$DEST/indococo_1v1"; cp "$2" "$DEST/indococo_1v1/1v1.pt"; fi
if [[ -n ${3:-} ]]; then mkdir -p "$DEST/indococo_430M"; cp "$3" "$DEST/indococo_430M/policy.safetensors"; fi
echo "pool bots in $DEST"
