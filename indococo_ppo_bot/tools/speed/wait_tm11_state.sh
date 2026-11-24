#!/bin/bash
# Compatibility wrapper. Prefer wait_tm_state.sh. Detects any team trainer.
exec "$(dirname "$0")/wait_tm_state.sh" "$@"
