#!/usr/bin/env bash
# Remove Ollama completely from the system (use sudo).
# Stops service, removes binary, service files, user/group, and data.
set -e
echo "Stopping and disabling Ollama service..."
sudo systemctl stop ollama 2>/dev/null || true
sudo systemctl disable ollama 2>/dev/null || true
echo "Removing Ollama binary and service files..."
sudo rm -f /usr/local/bin/ollama
sudo rm -f /etc/systemd/system/ollama.service
sudo rm -rf /etc/systemd/system/ollama.service.d
sudo systemctl daemon-reload 2>/dev/null || true
echo "Removing Ollama user and group (if present)..."
sudo userdel ollama 2>/dev/null || true
sudo groupdel ollama 2>/dev/null || true
echo "Removing Ollama data and config..."
rm -rf ~/.ollama
sudo rm -rf /etc/ollama
sudo rm -rf /var/log/ollama
sudo rm -rf /usr/share/ollama
echo "Ollama removed."
