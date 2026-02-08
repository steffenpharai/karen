#!/usr/bin/env bash
# Add CUDA to system PATH and LD_LIBRARY_PATH for all users. Run with sudo.
set -e
CUDA_DIR="/usr/local/cuda"
PROFILE="/etc/profile.d/cuda.sh"
echo "Creating $PROFILE with CUDA paths..."
cat > "$PROFILE" << 'EOF'
# CUDA (added by Jarvis install-cuda-path.sh)
if [ -d /usr/local/cuda/bin ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi
if [ -d /usr/local/cuda/lib64 ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
EOF
chmod 644 "$PROFILE"
echo "Done. Source with: . $PROFILE  (or log in again)"
. "$PROFILE"
nvcc --version
