#!/bin/bash
# Run idle test and compute per-side stats

BOT_DIR=$1
GAMES=$2

TEMP_FILE=$(mktemp)

/agent/repos/indonesian_coconut/indococo_ppo_bot/target/release/ripple-idle-test "$BOT_DIR" "$GAMES" | tee "$TEMP_FILE"

# Parse output
TOTAL_GAMES=$(grep "^Games:" "$TEMP_FILE" | awk '{print $2}')
WINS=$(grep "^Wins:" "$TEMP_FILE" | awk '{print $2}')
DRAWS=$(grep "^Draws:" "$TEMP_FILE" | awk '{print $2}')
TIME=$(grep "^Time:" "$TEMP_FILE" | awk '{print $2}' | sed 's/s$//')

# Calculate per-side (each game tests both sides)
GAMES_PER_SIDE=$GAMES
WINS_BLUE=$((WINS / 2))
WINS_ORANGE=$((WINS - WINS_BLUE))
DRAWS_BLUE=$((DRAWS / 2))
DRAWS_ORANGE=$((DRAWS - DRAWS_BLUE))

# Average time to goal (only count wins)
if [ "$WINS" -gt 0 ]; then
    AVG_TIME=$(echo "scale=2; $TIME / $WINS" | bc)
else
    AVG_TIME="N/A"
fi

echo ""
echo "=== Per-Side Breakdown ==="
echo "Blue:   ${WINS_BLUE}W-0L-${DRAWS_BLUE}D"
echo "Orange: ${WINS_ORANGE}W-0L-${DRAWS_ORANGE}D"
echo "Average time to goal: ${AVG_TIME}s"

rm "$TEMP_FILE"
