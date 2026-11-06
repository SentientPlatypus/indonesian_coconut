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

# RLBot v5 botpack (github.com/RLBot/botpack @ ef7fb02), commits as pinned there.
clone https://github.com/Rolv-Arild/Ripple-RLBot Ripple-RLBot 7ca4273a891ab1d04d001b48cb86f1bc8914aa85
clone https://github.com/stepkrav06/Kira Kira ba73b0557c4fa47b0408d09a5dd6b55059c59f58
clone https://github.com/V1XT1/Dice-BotPackV5 Dice-BotPackV5 88198294fa16bbee617fe38b3cf6e5ebb30e4569
clone https://github.com/KingJake3113/Chronos Chronos 4f8fdd27e4802d977b33b9ade4759ff6aeaca5aa
clone https://github.com/SoldierDoge/GalactusV2 GalactusV2 217324c2fc516048effe0407595fa61c2ebc6863
clone https://github.com/SubparN0va/ByteBotpack ByteBotpack HEAD

mkdir -p "$DEST"/{nexto,necto,element,immortal,ripple,kira,dice,chronos/rlbot,kira/rlbot,dice/rlbot,galactus/rlbot,megabyte}
cp "$SRC"/NectoFamily/nexto/{agent.py,nexto_obs.py,nexto-model.pt} "$DEST/nexto/"
cp "$SRC"/NectoFamily/necto/{agent.py,necto_obs.py,necto-model.pt} "$DEST/necto/"
cp -r "$SRC"/Elementv5/src/{agent.py,obs.py,model.p,sequences} "$DEST/element/"
cp -r "$SRC"/rlbotpack_v4/RLBotPack/Immortal/{agent.py,jit.pt,obs,action} "$DEST/immortal/"
mkdir -p "$DEST/seer"; cp "$SRC"/rlbotpack_v4/RLBotPack/Seer/{helper.py,Seer.pt} "$DEST/seer/"  # needs numba, scikit-learn

cp -r "$SRC"/Ripple-RLBot/{src,bots,LICENSE,NOTICE} "$DEST/ripple/"
for b in kira:Kira dice:Dice-BotPackV5 chronos:Chronos galactus:GalactusV2; do
  cp "$SRC/${b#*:}"/rlbot/{POLICY.lt,SHARED_HEAD.lt} "$DEST/${b%%:*}/rlbot/"
done
cp "$SRC"/ByteBotpack/rlbot/models/megabyte/{POLICY.lt,SHARED_HEAD.lt} "$DEST/megabyte/"

if [[ -n ${2:-} ]]; then mkdir -p "$DEST/indococo_1v1"; cp "$2" "$DEST/indococo_1v1/1v1.pt"; fi
if [[ -n ${3:-} ]]; then mkdir -p "$DEST/indococo_430M"; cp "$3" "$DEST/indococo_430M/policy.safetensors"; fi
echo "pool bots in $DEST"
