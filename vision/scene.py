"""Describe scene/motion/threat for LLM from YOLO + MediaPipe + depth + vitals + tracking.

Produces both a basic description (backward-compatible) and an enriched
description including tracked objects with depth, vitals, threat level,
object velocities, trajectories, ego-motion, and collision warnings.

The enriched description is injected into the LLM context so Jarvis can
make statements like:
  "Sir, person walking toward you at 1.2 m/s, 3.8 m away, potential
   collision in 2.4 seconds."
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


def describe_scene_with_perception(
    yolo_detections: list,
    face_count: int = 0,
    class_names: tuple | dict[int, str] | None = None,
    tracked_objects: list | None = None,
    depth_values: list | None = None,
    vitals=None,
    threat=None,
    perception_result=None,
) -> str:
    """Build the richest possible scene description using full perception data.

    Extends ``describe_scene_enriched`` with:
      - Per-object velocities in m/s (from flow + depth fusion)
      - Ego-motion status (walking, panning, static)
      - Trajectory predictions and collision alerts
      - Object behaviour classification (approaching, crossing, etc.)

    Parameters
    ----------
    perception_result : PerceptionResult from the perception pipeline, or None
    (all other params same as describe_scene_enriched)

    Example output:
      "Camera: walking forward. 2 people (1 approaching at 1.2m/s, ~3.8m,
       collision 2.4s), laptop, cup(2). [WARNING] Person from left at 4km/h.
       Vitals: mild fatigue. Threat: 3/10 low."
    """
    if class_names is None:
        class_names = COCO_NAMES

    parts = []

    # ── Ego-motion status ─────────────────────────────────────────
    if perception_result is not None:
        ego = getattr(perception_result, "ego_motion", None)
        if ego is not None and ego.is_moving:
            parts.append(f"Camera: {ego.motion_type}")

    # ── Objects with tracking + velocity + depth ──────────────────
    if tracked_objects and perception_result is not None:
        obj_descs = []
        seen_counts: dict[str, int] = {}
        detail_parts: list[str] = []

        trajs = getattr(perception_result, "trajectories", [])
        vel_mps = getattr(perception_result, "object_velocities_mps", [])

        for idx, t in enumerate(tracked_objects):
            name = getattr(t, "class_name", "") or _resolve_class_name(
                getattr(t, "cls", 0), class_names
            )
            seen_counts[name] = seen_counts.get(name, 0) + 1

            # Find trajectory info for this track
            traj = None
            if idx < len(trajs):
                traj = trajs[idx]

            v = vel_mps[idx] if idx < len(vel_mps) else None

            if traj and traj.behaviour != "stationary":
                speed_str = f"{v[2]:.1f}m/s" if v else ""
                depth_str = f"~{traj.depth_m:.1f}m" if traj.depth_m else ""
                ttc_str = (
                    f"collision {traj.time_to_collision:.1f}s"
                    if traj.time_to_collision
                    else ""
                )
                info_parts = [s for s in [
                    traj.behaviour, speed_str, depth_str, ttc_str
                ] if s]
                if info_parts:
                    detail_parts.append(f"{name} ({', '.join(info_parts)})")
            elif t.depth is not None:
                dist_m = t.depth * 10.0
                detail_parts.append(f"{name} (~{dist_m:.1f}m)")

        # Summary counts
        for name, count in sorted(seen_counts.items()):
            obj_descs.append(f"{name}({count})" if count > 1 else name)

        obj_text = ", ".join(obj_descs)
        if detail_parts:
            obj_text += " — " + "; ".join(detail_parts[:4])
        parts.append(obj_text)
    elif tracked_objects:
        # Fallback to basic enriched (no perception data)
        return describe_scene_enriched(
            yolo_detections, face_count, class_names,
            tracked_objects, depth_values, vitals, threat,
        )
    elif yolo_detections:
        seen = {}
        for d in yolo_detections:
            cls_id = d.get("cls", 0)
            name = _resolve_class_name(cls_id, class_names)
            seen[name] = seen.get(name, 0) + 1
        parts.append(", ".join(f"{n}({c})" if c > 1 else n for n, c in sorted(seen.items())))

    if face_count > 0:
        parts.append(f"{face_count} face(s)")

    # ── Collision alerts ──────────────────────────────────────────
    if perception_result is not None:
        alerts = getattr(perception_result, "collision_alerts", [])
        for alert in alerts[:2]:
            parts.append(f"[{alert.severity.upper()}] {alert.message}")

    # ── Vitals ────────────────────────────────────────────────────
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

    # ── Threat ────────────────────────────────────────────────────
    if threat is not None:
        level = getattr(threat, "level", 0)
        label = getattr(threat, "label", "clear")
        parts.append(f"Threat: {level}/10 {label}")

    if not parts:
        return "No notable objects"

    return ". ".join(parts)
