#!/bin/sh
# Spanish drill. Models are cached on disk after the first run.
#
#   ./run.sh                       drill
#   ./run.sh --model small         faster, less accurate
#   ./run.sh --alone               without serving it to anything else
#   ./run.sh --headless            serve with no window (no microphone)
#   ./run.sh --review              re-check past answers
#   ./run.sh --devices             list microphones
cd "$(dirname "$0")"

# The shared database, if this machine has been given the key. Both files are
# local and gitignored. Without them the drill behaves exactly as it always
# has, keeping its schedule in progress.json and talking to nobody.
if [ -f .sync-url ] && [ -f .sync-token ]; then
  SPANISH_DRILL_SYNC_URL="$(cat .sync-url)"
  SPANISH_DRILL_SYNC_TOKEN="$(cat .sync-token)"
  export SPANISH_DRILL_SYNC_URL SPANISH_DRILL_SYNC_TOKEN
fi

exec .venv/bin/python -m spanish_drill "$@"
