#!/usr/bin/env bash
# Free GPU/RAM on Jetson so Ollama can load the model on GPU (unified memory).
# Run with: sudo scripts/prepare-ollama-gpu.sh
# Then: bash scripts/start-ollama.sh (or ollama serve)
set -e
echo "Preparing Jetson for Ollama GPU..."
if command -v jetson_clocks &>/dev/null; then
  jetson_clocks && echo "jetson_clocks applied"
fi
# Drop disk caches to free RAM (helps unified memory)
if [ -w /proc/sys/vm/drop_caches ]; then
  echo 3 > /proc/sys/vm/drop_caches
  echo "Dropped caches"
fi
sync
echo "Done. Start Ollama with: bash scripts/start-ollama.sh (or ollama serve)"
