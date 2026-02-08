"""Local tools callable by the orchestrator LLM (vision, status, reminders, joke, sarcasm)."""

import logging
import random
from datetime import datetime
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# Synonym map for vision_analyze prompt (COCO class names)
_PROMPT_SYNONYMS = {
    "coffee mug": "cup",
    "mug": "cup",
    "coffee": "cup",
    "laptop computer": "laptop",
    "mobile": "cell phone",
    "phone": "cell phone",
    "tv": "tv",
    "television": "tv",
    "sofa": "couch",
    "dining table": "dining table",
}

# Lazy-loaded YOLO engine and class names (reused across vision_analyze calls)
_yolo_engine = None
_yolo_class_names = None


def _get_yolo():
    """Load and cache YOLO engine and class names."""
    global _yolo_engine, _yolo_class_names
    if _yolo_engine is None and settings.yolo_engine_exists():
        from vision.detector_yolo import get_class_names, load_yolo_engine

        _yolo_engine = load_yolo_engine(settings.YOLOE_ENGINE_PATH)
        _yolo_class_names = get_class_names(_yolo_engine) if _yolo_engine else None
    return _yolo_engine, _yolo_class_names


def vision_analyze(prompt: str | None = None) -> str:
    """Run vision on current camera frame; optional prompt to focus (e.g. person, cup). Returns objects, counts, brief description."""
    try:
        from vision.camera import open_camera, read_frame
        from vision.detector_mediapipe import create_face_detector, detect_faces
        from vision.detector_yolo import run_inference
        from vision.scene import describe_scene

        cap = open_camera(
            settings.CAMERA_INDEX,
            settings.CAMERA_WIDTH,
            settings.CAMERA_HEIGHT,
            settings.CAMERA_FPS,
            device_path=settings.CAMERA_DEVICE,
        )
        if not cap:
            return "Vision temporarily unavailable (camera not found)."
        try:
            frame = read_frame(cap)
        finally:
            cap.release()
        if frame is None:
            return "No frame captured."
        engine, class_names = _get_yolo()
        if not engine:
            return "Vision temporarily unavailable (engine not loaded)."
        yolo_dets = run_inference(engine, frame)
        face_det = create_face_detector()
        faces = detect_faces(face_det, frame) if face_det else []
        base_desc = describe_scene(yolo_dets, face_count=len(faces), class_names=class_names)
        if not prompt or not prompt.strip():
            return f"Objects: {base_desc}. Face count: {len(faces)}."
        q = prompt.strip().lower()
        focus = _PROMPT_SYNONYMS.get(q, q)
        if focus in base_desc.lower():
            return f"Objects: {base_desc}. Face count: {len(faces)}. Note: '{focus}' detected."
        from vision.scene import COCO_NAMES

        if focus in [c.lower() for c in COCO_NAMES]:
            return f"Objects: {base_desc}. Face count: {len(faces)}. Note: '{focus}' detected."
        return f"Objects: {base_desc}. Face count: {len(faces)}."
    except Exception as e:
        logger.warning("vision_analyze failed: %s", e)
        return "Vision temporarily unavailable."


def get_jetson_status() -> str:
    """Return GPU/mem/temp/power and mode for the Jetson (jtop or tegrastats)."""
    try:
        from utils.power import get_system_stats, get_thermal_warning

        stats = get_system_stats()
        thermal = get_thermal_warning()
        parts = [stats] if stats else []
        if thermal:
            parts.append(thermal)
        return "; ".join(parts) if parts else "System stats unavailable."
    except Exception as e:
        logger.warning("get_jetson_status failed: %s", e)
        return "System stats unavailable."


def get_current_time() -> str:
    """Return formatted current time and date."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_reminder(text: str, time_str: str = "") -> str:
    """Save a reminder to the JSON file. time_str is optional (e.g. '14:00' or 'tomorrow')."""
    try:
        from utils.reminders import add_reminder

        data_dir = Path(settings.DATA_DIR)
        add_reminder(data_dir, text, time_str)
        return f"Reminder added: {text}" + (f" at {time_str}" if time_str else "") + "."
    except Exception as e:
        logger.warning("create_reminder failed: %s", e)
        return "Failed to add reminder."


def list_reminders() -> str:
    """Read and summarize pending reminders."""
    try:
        from utils.reminders import format_reminders_for_llm, load_reminders

        data_dir = Path(settings.DATA_DIR)
        reminders = load_reminders(data_dir)
        out = format_reminders_for_llm(reminders, max_items=10)
        return out if out else "No pending reminders."
    except Exception as e:
        logger.warning("list_reminders failed: %s", e)
        return "Could not list reminders."


_JOKES = [
    "I would avoid the priesthood, Sir. The only thing they're good at is wine and wafer management.",
    "I've calculated the odds of your survival. I'd rather not share them.",
    "Shall I alert the press that the great Tony Stark has misplaced his keys?",
    "Your security protocol appears to be 'hope for the best'. Charming.",
    "I'm afraid the only thing unbreakable in this scenario is my patience.",
]


def tell_joke() -> str:
    """Return a random dry/witty one-liner."""
    return random.choice(_JOKES)


def toggle_sarcasm(enabled: bool) -> str:
    """Turn sarcasm mode on or off. Returns confirmation."""
    settings.SARCASM_ENABLED = bool(enabled)
    return "Sarcasm mode on." if enabled else "Sarcasm mode off."


# Ollama-compatible tool schemas (list of dicts for API tools payload)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "vision_analyze",
            "description": "Run vision on current camera frame; optional prompt to focus (e.g. person, cup). Returns objects, counts, brief description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Optional focus, e.g. 'person', 'coffee mug'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_jetson_status",
            "description": "Get Jetson GPU/memory/temperature/power and power mode (tegrastats/jtop).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current date and time formatted.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Save a reminder. Optional time (e.g. '14:00' or 'tomorrow').",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Reminder text."},
                    "time_str": {"type": "string", "description": "Optional time."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List pending reminders.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tell_joke",
            "description": "Tell a dry, witty one-liner.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_sarcasm",
            "description": "Turn sarcasm mode on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "description": "True to enable sarcasm."},
                },
                "required": ["enabled"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "vision_analyze": vision_analyze,
    "get_jetson_status": get_jetson_status,
    "get_current_time": get_current_time,
    "create_reminder": create_reminder,
    "list_reminders": list_reminders,
    "tell_joke": tell_joke,
    "toggle_sarcasm": toggle_sarcasm,
}


def run_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with given arguments; return string result."""
    fn = TOOL_REGISTRY.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        # Ollama may send "time" for create_reminder
        if name == "create_reminder" and "time" in arguments and "time_str" not in arguments:
            arguments = {**arguments, "time_str": arguments.get("time", "")}
        # Only pass known params to each tool
        if name == "vision_analyze":
            args = {"prompt": arguments.get("prompt")}
        elif name == "create_reminder":
            args = {"text": arguments.get("text", ""), "time_str": arguments.get("time_str", arguments.get("time", ""))}
        elif name == "toggle_sarcasm":
            args = {"enabled": arguments.get("enabled", False)}
        else:
            args = {}
        result = fn(**args)
        return str(result)
    except Exception as e:
        logger.warning("run_tool %s failed: %s", name, e)
        return f"Tool error: {e}"
