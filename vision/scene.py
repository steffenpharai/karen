"""Describe scene/motion/threat for LLM from YOLO + MediaPipe + depth + vitals + tracking.

Produces both a basic description (backward-compatible) and an enriched
description including tracked objects with depth, vitals, and threat level.
"""

import math

# COCO class names (YOLO default)
COCO_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def _resolve_class_name(cls_id: int, class_names) -> str:
    """Resolve a class ID to a name using the provided names mapping."""
    if isinstance(class_names, dict):
        return class_names.get(cls_id, f"class_{cls_id}")
    if isinstance(class_names, (tuple, list)):
        return class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
    return f"class_{cls_id}"


def describe_scene(
    yolo_detections: list,
    face_count: int = 0,
    class_names: tuple | dict[int, str] | None = None,
) -> str:
    """Build a short scene description for the LLM (backward-compatible).

    class_names can be tuple (COCO), dict (e.g. YOLOE), or None for COCO.
    """
    if class_names is None:
        class_names = COCO_NAMES
    parts = []
    if yolo_detections:
        seen = {}
        for d in yolo_detections:
            cls_id = d.get("cls", 0)
            name = _resolve_class_name(cls_id, class_names)
            seen[name] = seen.get(name, 0) + 1
        parts.append(", ".join(f"{n}({c})" for n, c in sorted(seen.items())))
    if face_count > 0:
        parts.append(f"{face_count} face(s)")
    return "; ".join(parts) if parts else "No notable objects"


def describe_scene_enriched(
    yolo_detections: list,
    face_count: int = 0,
    class_names: tuple | dict[int, str] | None = None,
    tracked_objects: list | None = None,
    depth_values: list | None = None,
    vitals=None,
    threat=None,
) -> str:
    """Build a rich scene description with tracking, depth, vitals, and threat data.

    Example output:
      "2 people (1 approaching, ~3m), laptop, cup(2). Vitals: mild fatigue,
       posture fair. Threat: 2/10 clear."

    Parameters
    ----------
    yolo_detections : raw YOLOE detections
    face_count : number of faces detected
    class_names : class ID -> name mapping
    tracked_objects : list of TrackedObject from ByteTrackLite
    depth_values : per-detection depth (0-1) from depth_at_boxes
    vitals : VitalsResult from VitalsAnalyzer
    threat : ThreatAssessment from ThreatScorer
    """
    if class_names is None:
        class_names = COCO_NAMES

    parts = []

    # ── Objects with tracking + depth ─────────────────────────────────
    if tracked_objects:
        obj_descs = []
        seen_counts: dict[str, int] = {}
        approaching: list[str] = []

        for t in tracked_objects:
            name = getattr(t, "class_name", "") or _resolve_class_name(
                getattr(t, "cls", 0), class_names
            )
            seen_counts[name] = seen_counts.get(name, 0) + 1

            # Approaching detection
            speed = math.sqrt(t.velocity[0] ** 2 + t.velocity[1] ** 2)
            if speed > 30 and name == "person":
                depth_str = ""
                if t.depth is not None:
                    dist_m = t.depth * 10.0  # relative -> pseudo-meters
                    depth_str = f", ~{dist_m:.1f}m"
                approaching.append(f"1 approaching{depth_str}")

        for name, count in sorted(seen_counts.items()):
            obj_descs.append(f"{name}({count})" if count > 1 else name)

        obj_text = ", ".join(obj_descs)
        if approaching:
            obj_text += f" ({'; '.join(approaching)})"
        parts.append(obj_text)
    elif yolo_detections:
        # Fallback to basic detection description
        seen = {}
        for i, d in enumerate(yolo_detections):
            cls_id = d.get("cls", 0)
            name = _resolve_class_name(cls_id, class_names)
            # Add depth info if available
            if depth_values and i < len(depth_values) and depth_values[i] is not None:
                dist_m = depth_values[i] * 10.0
                name_depth = f"{name}(~{dist_m:.1f}m)"
            else:
                name_depth = name
            seen[name_depth] = seen.get(name_depth, 0) + 1
        parts.append(", ".join(f"{n}({c})" if c > 1 else n for n, c in sorted(seen.items())))

    if face_count > 0:
        parts.append(f"{face_count} face(s)")

    # ── Vitals ────────────────────────────────────────────────────────
    if vitals is not None:
        vitals_parts = []
        fatigue = getattr(vitals, "fatigue_level", "unknown")
        if fatigue != "unknown":
            vitals_parts.append(f"{fatigue} fatigue")
        posture = getattr(vitals, "posture_label", "unknown")
        if posture != "unknown":
            vitals_parts.append(f"posture {posture}")
        hr = getattr(vitals, "heart_rate_bpm", None)
        if hr is not None:
            conf = getattr(vitals, "heart_rate_confidence", 0)
            vitals_parts.append(f"HR ~{hr:.0f}bpm" + ("?" if conf < 0.5 else ""))
        if vitals_parts:
            parts.append("Vitals: " + ", ".join(vitals_parts))

    # ── Threat ────────────────────────────────────────────────────────
    if threat is not None:
        level = getattr(threat, "level", 0)
        label = getattr(threat, "label", "clear")
        parts.append(f"Threat: {level}/10 {label}")

    if not parts:
        return "No notable objects"

    return ". ".join(parts)
