#!/usr/bin/env bash
# Install cuSPARSELt for PyTorch on Jetson (CUDA 12.5–12.9). Run with sudo.
# Ref: https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cusparselt.sh
set -e
CUDA_VERSION="${CUDA_VERSION:-12.6}"
# aarch64 -> sbsa for NVIDIA redist
arch_path='sbsa'
TARGETARCH="${TARGETARCH:-$(uname -m)}"
if [ "$TARGETARCH" = 'x86_64' ] || [ "$TARGETARCH" = 'amd64' ]; then
  arch_path='x86_64'
fi
CUSPARSELT_NAME="libcusparse_lt-linux-${arch_path}-0.7.1.0-archive"
URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-${arch_path}/${CUSPARSELT_NAME}.tar.xz"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local/cuda}"
echo "Installing cuSPARSELt for CUDA ${CUDA_VERSION} (${arch_path}) into ${INSTALL_PREFIX}..."
mkdir -p /tmp/cusparselt_install && cd /tmp/cusparselt_install
curl --retry 3 -OLs "$URL"
tar xf "${CUSPARSELT_NAME}.tar.xz"
sudo cp -a "${CUSPARSELT_NAME}/include/"* "${INSTALL_PREFIX}/include/" 2>/dev/null || true
sudo cp -a "${CUSPARSELT_NAME}/lib/"* "${INSTALL_PREFIX}/lib64/"
sudo ldconfig 2>/dev/null || true
cd ..
rm -rf /tmp/cusparselt_install
echo "Done. Ensure LD_LIBRARY_PATH includes ${INSTALL_PREFIX}/lib64 (e.g. via /etc/profile.d/cuda.sh)."
