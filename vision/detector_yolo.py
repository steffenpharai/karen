"""YOLOE-26N TensorRT .engine inference on Orin (2026 Ultralytics YOLOE)."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_yolo_engine(engine_path: str | Path):
    """Load YOLOE TensorRT engine. Returns model or None. Uses YOLO() for .engine (TensorRT backend)."""
    path = Path(engine_path)
    if not path.exists():
        logger.warning("YOLOE engine not found: %s", path)
        return None
    try:
        from ultralytics import YOLO

        return YOLO(str(path))
    except Exception as e:
        logger.warning("YOLOE engine load failed: %s", e)
        return None


def get_class_names(model) -> dict[int, str] | None:
    """Return class index -> name mapping from model (e.g. YOLOE prompt-free 4585 classes), or None."""
    if model is None:
        return None
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)
    if isinstance(names, (list, tuple)):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return names
    return None


def run_inference(model, frame) -> list:
    """Run detection on frame; return list of detections (xyxy, conf, cls)."""
    if model is None or frame is None:
        return []
    try:
        results = model(frame, verbose=False)
        out = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    out.append({"xyxy": xyxy, "conf": conf, "cls": cls_id})
        return out
    except Exception as e:
        logger.warning("YOLO inference failed: %s", e)
        return []
