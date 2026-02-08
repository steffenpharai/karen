#!/usr/bin/env bash
# Install Ollama locally on Jetson — exactly per Jetson AI Lab native install.
# https://www.jetson-ai-lab.com/tutorials/ollama/
# Prerequisites: JetPack 5 (r35.x) or JetPack 6 (r36.x); NVMe SSD recommended.
set -e
echo "Installing Ollama (Jetson AI Lab native install)..."
curl -fsSL https://ollama.com/install.sh | sh
echo "Done. The installer creates a service to run 'ollama serve' on startup."
echo "To run a model: ollama run llama3.2:1b   (or smaller if OOM)"
echo "For GPU on Jetson, ensure ollama service uses GPU; if needed run: bash scripts/start-ollama.sh"
echo "Memory: If OOM, use smaller models. List: https://ollama.com/library"
