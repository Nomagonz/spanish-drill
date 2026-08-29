#!/bin/sh
# Local Spanish drill. First run downloads the Whisper model (~500MB for "small").
cd "$(dirname "$0")"
exec .venv/bin/python drill.py "$@"
