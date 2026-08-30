#!/bin/sh
# Spanish drill. Models are cached on disk after the first run.
#
#   ./run.sh                       drill
#   ./run.sh --model small         faster, less accurate
#   ./run.sh --review              re-check past answers
#   ./run.sh --devices             list microphones
cd "$(dirname "$0")"
exec .venv/bin/python -m spanish_drill "$@"
