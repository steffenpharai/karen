#!/usr/bin/env bash
# Configure systemd ollama service for GPU on Jetson: OLLAMA_NUM_GPU=1 and LimitMEMLOCK=infinity.
# Run with: sudo scripts/configure-ollama-systemd.sh
# Then: sudo systemctl daemon-reload && sudo systemctl restart ollama
set -e
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/gpu.conf << 'EOF'
[Service]
Environment="OLLAMA_NUM_GPU=1"
LimitMEMLOCK=infinity
EOF
echo "Created /etc/systemd/system/ollama.service.d/gpu.conf"
echo "Run: sudo systemctl daemon-reload && sudo systemctl restart ollama"
