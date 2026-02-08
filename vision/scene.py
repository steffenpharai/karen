"""Describe scene/motion/threat for LLM from YOLO + MediaPipe."""

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


def describe_scene(
    yolo_detections: list,
    face_count: int = 0,
    class_names: tuple | dict[int, str] | None = None,
) -> str:
    """Build a short scene description for the LLM. class_names can be tuple (COCO), dict (e.g. YOLOE), or None for COCO."""
    if class_names is None:
        class_names = COCO_NAMES
    parts = []
    if yolo_detections:
        seen = {}
        for d in yolo_detections:
            cls_id = d.get("cls", 0)
            if isinstance(class_names, dict):
                name = class_names.get(cls_id, f"class_{cls_id}")
            else:
                name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            seen[name] = seen.get(name, 0) + 1
        parts.append(", ".join(f"{n}({c})" for n, c in sorted(seen.items())))
    if face_count > 0:
        parts.append(f"{face_count} face(s)")
    return "; ".join(parts) if parts else "No notable objects"
