#!/usr/bin/env bash
# Start Ollama server locally with GPU (Jetson). Per Jetson AI Lab native install.
# https://www.jetson-ai-lab.com/tutorials/ollama/
# Install first: bash scripts/install-ollama.sh
# Use this script to run Ollama manually with GPU (e.g. if the system service isn't using GPU).
set -e

if ! command -v ollama &>/dev/null; then
  echo "Ollama not found. Install with: bash scripts/install-ollama.sh"
  exit 1
fi

# Free GPU/RAM on Jetson (unified memory) so the model can load. Run with sudo if needed.
if command -v jetson_clocks &>/dev/null; then
  if jetson_clocks &>/dev/null; then
    echo "jetson_clocks applied"
  else
    echo "Tip: run 'sudo jetson_clocks' before starting Ollama to free GPU memory"
  fi
fi

# Run on GPU. API at http://127.0.0.1:11434
export OLLAMA_NUM_GPU=1
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
echo "Starting Ollama (OLLAMA_NUM_GPU=1). API at http://${OLLAMA_HOST}"
exec ollama serve
