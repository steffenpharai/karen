"""Paths, model names, RAM/thermal limits for Jetson Orin Nano 8GB."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_ROOT = os.path.join(PROJECT_ROOT, "venv")

# RAM budget (GiB) – keep under 7.5 to avoid swap on microSD
RAM_BUDGET_GIB = 7.5

# Ollama – local install, GPU. Default llama3.2:1b (8GB-friendly). Override with OLLAMA_MODEL.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_FALLBACK_MODEL = os.environ.get("OLLAMA_FALLBACK_MODEL", "llama3.2:1b")  # same or smaller on OOM
# Context size (KV cache). Lower = less GPU RAM; 1024 fits 8GB Jetson when model loads.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "1024"))

# Voice
WAKE_WORD_MODEL = "jarvis"  # openWakeWord / Porcupine
STT_MODEL_SIZE = "small"  # faster-whisper: tiny, base, small, medium
# Piper British male: path to .onnx (default from models/voices/)
_DEFAULT_TTS_ONNX = os.path.join(PROJECT_ROOT, "models", "voices", "en_GB-alan-medium.onnx")
TTS_VOICE = os.environ.get("JARVIS_TTS_VOICE", _DEFAULT_TTS_ONNX)
RECORD_DURATION_SEC = 5.0  # seconds to record after wake
SARCASM_ENABLED = False  # toggle for dry/sarcastic replies

# Vision – USB camera (e.g. /dev/video0 on Jetson). Use index 0 by default.
# Set JARVIS_CAMERA_DEVICE=/dev/video0 to force a device path.
CAMERA_DEVICE = os.environ.get("JARVIS_CAMERA_DEVICE")  # None = use index
CAMERA_INDEX = int(os.environ.get("JARVIS_CAMERA_INDEX", "0"))
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
# YOLOE-26N (2026): prompt-free nano for Jetson 8GB; engine built from yoloe-26n-seg-pf.pt
YOLOE_ENGINE_PATH = os.path.join(PROJECT_ROOT, "models", "yoloe26n.engine")
# Legacy alias for backward compatibility
YOLO_ENGINE_PATH = YOLOE_ENGINE_PATH


def yolo_engine_exists() -> bool:
    """True if the TensorRT YOLOE engine file exists (required for vision pipeline)."""
    return os.path.isfile(YOLOE_ENGINE_PATH)


# Optional GUI vision preview (Phase 5)
JARVIS_PREVIEW_PATH = os.path.join(os.environ.get("TMPDIR", "/tmp"), "jarvis_preview.jpg")
