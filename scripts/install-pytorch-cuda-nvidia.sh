#!/usr/bin/env bash
# Install PyTorch with CUDA on Jetson Orin Nano per NVIDIA / community (current as of 2026).
#
# Where to get wheels (2026):
#   • Primary: NVIDIA forum sticky (updated by NVIDIA/dusty_nv):
#     https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
#     (Look for JetPack 6.x section; PyTorch 2.3–2.8.)
#   • Jetson AI Lab index (JetPack 6.2 + CUDA 12.6):
#     https://pypi.jetson-ai-lab.dev/jp6/cu126/
#     pip install torch torchvision --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu126/
#   • Official NVIDIA redist (sometimes older):
#     https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/ (or v60)
#
# Prereqs: (1) sudo apt-get -y update; sudo apt-get install -y python3-pip libopenblas-dev
#          (2) For 24.06+: install cuSPARSELt (sudo bash scripts/install-cusparselt.sh)
# Run from project root with venv activated. Source /etc/profile.d/cuda.sh for LD_LIBRARY_PATH.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Prefer Jetson AI Lab index for JetPack 6.2 + CUDA 12.6 (set to 0 to use NVIDIA redist only)
USE_JETSON_AI_LAB="${USE_JETSON_AI_LAB:-1}"
JETSON_AI_LAB_INDEX="https://pypi.jetson-ai-lab.dev/jp6/cu126/"

# NVIDIA official redist fallback (JetPack 6.0/6.1; v61)
JP_VERSION="${JP_VERSION:-61}"
TORCH_WHEEL="torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
TORCH_INSTALL="https://developer.download.nvidia.com/compute/redist/jp/v${JP_VERSION}/pytorch/${TORCH_WHEEL}"
# TorchVision for torch 2.5 (Ultralytics assets; compatible with JP 6.x)
TORCHVISION_WHEEL="torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl"
TORCHVISION_INSTALL="https://github.com/ultralytics/assets/releases/download/v0.0.0/${TORCHVISION_WHEEL}"

pip install --upgrade pip
pip install "numpy==1.26.1"
pip uninstall -y torch torchvision 2>/dev/null || true

if [[ "$USE_JETSON_AI_LAB" == "1" ]]; then
  echo "Using Jetson AI Lab index (JetPack 6.2 + CUDA 12.6): $JETSON_AI_LAB_INDEX"
  if pip install --no-cache-dir torch torchvision --extra-index-url "$JETSON_AI_LAB_INDEX"; then
    echo "Jetson AI Lab install succeeded."
  else
    echo "Jetson AI Lab install failed; falling back to NVIDIA redist..."
    pip install --no-cache-dir "$TORCH_INSTALL"
    pip install --no-cache-dir "$TORCHVISION_INSTALL" || true
  fi
else
  echo "Using NVIDIA redist: $TORCH_INSTALL"
  pip install --no-cache-dir "$TORCH_INSTALL"
  pip install --no-cache-dir "$TORCHVISION_INSTALL" || pip install 'torchvision>=0.9.0' || true
fi

echo "Done. Verify with: . /etc/profile.d/cuda.sh && python -c \"import torch; print('CUDA:', torch.cuda.is_available()); import torchvision; print('torchvision:', torchvision.__version__)\""
