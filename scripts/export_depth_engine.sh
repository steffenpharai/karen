#!/usr/bin/env bash
# Export DepthAnything V2 Small to TensorRT FP16 engine on Jetson Orin Nano.
#
# Prerequisites:
#   - Python 3.10 venv with torch, onnx, onnxruntime
#   - TensorRT (comes with JetPack 6.x)
#   - ~2 GB free disk space
#
# Usage:
#   bash scripts/export_depth_engine.sh
#
# Output: models/depth_anything_v2_small.engine

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_ROOT}/models"
ONNX_PATH="${MODELS_DIR}/depth_anything_v2_small.onnx"
ENGINE_PATH="${MODELS_DIR}/depth_anything_v2_small.engine"
INPUT_H=518
INPUT_W=518

mkdir -p "$MODELS_DIR"

echo "=== DepthAnything V2 Small → TensorRT FP16 ==="
echo "Project root: $PROJECT_ROOT"
echo "Output engine: $ENGINE_PATH"

# Step 1: Export to ONNX if not already done
if [ ! -f "$ONNX_PATH" ]; then
    echo ""
    echo "--- Step 1: Export to ONNX ---"
    python3 -c "
import torch
import sys

try:
    model = torch.hub.load(
        'LiheYoung/depth-anything',
        'DepthAnything_V2_Small',
        pretrained=True,
        trust_repo=True,
    )
    model.eval()

    dummy = torch.randn(1, 3, ${INPUT_H}, ${INPUT_W})
    torch.onnx.export(
        model,
        dummy,
        '${ONNX_PATH}',
        opset_version=17,
        input_names=['input'],
        output_names=['depth'],
        dynamic_axes=None,
    )
    print(f'ONNX exported: ${ONNX_PATH}')
except Exception as e:
    print(f'ONNX export failed: {e}', file=sys.stderr)
    print('You can manually download the ONNX model and place it at:', file=sys.stderr)
    print(f'  ${ONNX_PATH}', file=sys.stderr)
    sys.exit(1)
"
else
    echo "ONNX model already exists: $ONNX_PATH"
fi

# Step 2: Convert ONNX to TensorRT FP16 engine
if [ ! -f "$ENGINE_PATH" ]; then
    echo ""
    echo "--- Step 2: ONNX → TensorRT FP16 engine ---"
    echo "This may take 5-15 minutes on Jetson Orin Nano..."

    /usr/src/tensorrt/bin/trtexec \
        --onnx="$ONNX_PATH" \
        --saveEngine="$ENGINE_PATH" \
        --fp16 \
        --workspace=2048 \
        --verbose 2>&1 | tail -20

    if [ -f "$ENGINE_PATH" ]; then
        echo ""
        echo "Engine built successfully: $ENGINE_PATH"
        ls -lh "$ENGINE_PATH"
    else
        echo "ERROR: Engine build failed. Check trtexec output above."
        exit 1
    fi
else
    echo "TensorRT engine already exists: $ENGINE_PATH"
fi

echo ""
echo "=== Done. Set DEPTH_ENABLED=1 to enable depth estimation. ==="
echo "  export JARVIS_DEPTH_ENABLED=1"
