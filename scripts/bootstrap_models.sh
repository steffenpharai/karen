#!/usr/bin/env bash
# Download all Jarvis models: openWakeWord, Faster-Whisper, Piper voice; optionally build YOLOE-26N engine.
# Run from project root. Activate venv first, or this script will try to use ./venv.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -d "$PROJECT_ROOT/venv" ]]; then
  source "$PROJECT_ROOT/venv/bin/activate"
fi

python scripts/bootstrap_models.py "$@"
