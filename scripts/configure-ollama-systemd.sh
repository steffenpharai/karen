#!/usr/bin/env bash
# Configure systemd ollama service for GPU on Jetson: OLLAMA_NUM_GPU=1 and LimitMEMLOCK=infinity.
# Optionally set OLLAMA_NUM_CTX and OLLAMA_KEEP_ALIVE (for 8GB Jetson, match Jarvis app context).
# Run with: sudo scripts/configure-ollama-systemd.sh
# Optional: OLLAMA_NUM_CTX=1024 OLLAMA_KEEP_ALIVE=-1 sudo scripts/configure-ollama-systemd.sh
# Then: sudo systemctl daemon-reload && sudo systemctl restart ollama
set -e
mkdir -p /etc/systemd/system/ollama.service.d
NUM_CTX="${OLLAMA_NUM_CTX:-1024}"
KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
cat > /etc/systemd/system/ollama.service.d/gpu.conf << EOF
[Service]
Environment="OLLAMA_NUM_GPU=1"
Environment="OLLAMA_NUM_CTX=${NUM_CTX}"
Environment="OLLAMA_KEEP_ALIVE=${KEEP_ALIVE}"
Environment="OLLAMA_NUM_PARALLEL=1"
LimitMEMLOCK=infinity
EOF
echo "Created /etc/systemd/system/ollama.service.d/gpu.conf (OLLAMA_NUM_CTX=${NUM_CTX}, OLLAMA_KEEP_ALIVE=${KEEP_ALIVE})"
echo "Run: sudo systemctl daemon-reload && sudo systemctl restart ollama"
