"""YOLOE-26N TensorRT .engine inference on Orin (2026 Ultralytics YOLOE).

Supports both prompt-free (broad 4585-class) and dynamic open-vocabulary
prompting via ``set_classes()`` / post-inference filtering.
"""

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


# ── Dynamic open-vocabulary prompting ─────────────────────────────────


def _try_set_classes(model, classes: list[str]) -> bool:
    """Attempt ``model.set_classes(classes)`` for YOLOE open-vocab.

    Returns True if successful, False if the engine doesn't support it
    (e.g. exported TensorRT .engine without dynamic class embedding).
    """
    try:
        if hasattr(model, "set_classes"):
            model.set_classes(classes)
            return True
    except Exception as e:
        logger.debug("set_classes() not supported on this engine: %s", e)
    return False


def run_inference_with_classes(
    model,
    frame,
    classes: list[str],
    class_names: dict[int, str] | None = None,
    conf_threshold: float = 0.25,
) -> list:
    """Run detection focused on specific classes.

    Strategy:
    1. Try ``model.set_classes(classes)`` for true open-vocab prompting.
    2. If that fails (common with .engine files), run normal prompt-free
       inference and post-filter results to only matching class names.

    Parameters
    ----------
    model : YOLO model instance
    frame : numpy array (BGR)
    classes : list of class name strings, e.g. ["person", "coffee mug"]
    class_names : dict mapping cls_id -> name (from get_class_names)
    conf_threshold : minimum confidence to keep

    Returns
    -------
    list of detection dicts {xyxy, conf, cls, class_name}
    """
    if model is None or frame is None or not classes:
        return []

    # Normalise requested class names to lowercase for matching
    requested = {c.strip().lower() for c in classes if c.strip()}
    if not requested:
        return run_inference(model, frame)

    # Attempt native open-vocab prompting
    if _try_set_classes(model, list(requested)):
        dets = run_inference(model, frame)
        # set_classes remaps class IDs; trust the model's new names
        new_names = get_class_names(model)
        for d in dets:
            cid = d["cls"]
            if new_names and cid in new_names:
                d["class_name"] = new_names[cid]
        return [d for d in dets if d.get("conf", 0) >= conf_threshold]

    # Fallback: run prompt-free, then filter by class name
    dets = run_inference(model, frame)
    if class_names is None:
        return dets  # can't filter without names

    filtered = []
    for d in dets:
        cid = d["cls"]
        name = class_names.get(cid, f"class_{cid}").lower()
        d["class_name"] = name
        # Match if any requested term appears in the detection name or vice-versa
        if any(req in name or name in req for req in requested):
            if d.get("conf", 0) >= conf_threshold:
                filtered.append(d)
    return filtered
