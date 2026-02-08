#!/usr/bin/env bash
# Inspect effective Ollama configuration (systemd env and overrides).
# Run with sudo for systemd properties: sudo scripts/inspect-ollama-config.sh
# Without sudo, only user-writable paths and hints are shown.
set -e

echo "=== Ollama configuration (Jetson 8GB) ==="
echo ""

# Systemd override files (need sudo to read if restricted)
echo "--- Systemd override files ---"
if [ -d /etc/systemd/system/ollama.service.d ]; then
  for f in /etc/systemd/system/ollama.service.d/*.conf; do
    [ -f "$f" ] || continue
    echo "File: $f"
    cat "$f" 2>/dev/null || sudo cat "$f" 2>/dev/null || echo "(cannot read)"
    echo ""
  done
else
  echo "No /etc/systemd/system/ollama.service.d (Ollama may not be installed as systemd service)"
fi

# Effective environment (requires sudo when run as systemd service)
echo "--- Effective service environment ---"
if systemctl show ollama --property=Environment 2>/dev/null | grep -q Environment; then
  systemctl show ollama --property=Environment 2>/dev/null || sudo systemctl show ollama --property=Environment 2>/dev/null || true
else
  echo "Ollama service not found or not loaded. If using systemd: sudo systemctl show ollama --property=Environment"
fi
echo ""

# User/data paths
echo "--- Paths ---"
echo "Models (default): ${OLLAMA_MODELS:-$HOME/.ollama/models}"
[ -d "${HOME}/.ollama" ] && echo "  $HOME/.ollama exists" || true
[ -d /etc/ollama ] && echo "  /etc/ollama exists" || true
[ -d /var/log/ollama ] && echo "  /var/log/ollama exists" || true
echo ""

# Suggestions for 8GB Jetson
echo "--- Suggested for 8GB Jetson ---"
echo "To set context and keep-alive, add to systemd override (e.g. sudo scripts/configure-ollama-systemd.sh):"
echo '  Environment="OLLAMA_NUM_GPU=1"'
echo '  Environment="OLLAMA_NUM_CTX=1024"   # match Jarvis config/settings.py OLLAMA_NUM_CTX'
echo '  Environment="OLLAMA_KEEP_ALIVE=-1"  # keep model loaded; use 5m or 0 to free GPU when idle'
echo '  Environment="OLLAMA_NUM_PARALLEL=1" # avoid OOM'
echo ""
echo "Then: sudo systemctl daemon-reload && sudo systemctl restart ollama"
echo "Inspect again: sudo scripts/inspect-ollama-config.sh"
