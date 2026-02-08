#!/usr/bin/env bash
# Export YOLOE-26N (2026) to TensorRT .engine on Jetson Orin. Run from project root with venv active.
# Per NVIDIA: CUDA and LD_LIBRARY_PATH must be set (e.g. . /etc/profile.d/cuda.sh).
# Uses Ultralytics YOLOE prompt-free nano: yoloe-26n-seg-pf.pt (no prompts, 4585-class vocab).
set -e
[ -f /etc/profile.d/cuda.sh ] && . /etc/profile.d/cuda.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"
mkdir -p "$MODELS_DIR"
cd "$PROJECT_ROOT"

# Ensure ultralytics with YOLOE support
python -c "from ultralytics import YOLOE; print('YOLOE OK')"

# Require CUDA for production TensorRT export (no workarounds)
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: CUDA is required for YOLOE TensorRT export. Install PyTorch with CUDA for Jetson (see README)."
  exit 1
fi

# Text-prompt YOLOE (yoloe-26n-seg.pt) exports reliably; set_classes before export for fuse.
echo "Exporting YOLOE-26N (yoloe-26n-seg.pt) to TensorRT engine on device 0 (CUDA)..."
python -c "
from ultralytics import YOLOE
model = YOLOE('yoloe-26n-seg.pt')
# COCO-like set so fuse has text embeddings (required for TensorRT export)
model.set_classes(['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'])
model.export(format='engine', device=0, half=True, workspace=4)
"
ENGINE_SRC="yoloe-26n-seg.engine"
ENGINE_DST="$MODELS_DIR/yoloe26n.engine"
if [[ -f "$ENGINE_SRC" ]]; then
  mv -f "$ENGINE_SRC" "$ENGINE_DST"
  echo "Saved $ENGINE_DST"
  exit 0
fi
echo "Export did not produce $ENGINE_SRC"
exit 1
