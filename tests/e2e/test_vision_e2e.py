"""E2E tests for vision pipeline (camera, YOLOE, scene). Engine required for full vision E2E."""

import pytest
from config import settings
from vision import camera, detector_yolo, scene


@pytest.mark.e2e
def test_yolo_engine_exists_and_loads():
    """When engine exists at YOLOE_ENGINE_PATH, it must load (required for --e2e vision)."""
    if not settings.yolo_engine_exists():
        pytest.skip("models/yoloe26n.engine not found; run bash scripts/export_yolo_engine.sh")
    model = detector_yolo.load_yolo_engine(settings.YOLOE_ENGINE_PATH)
    assert model is not None


@pytest.mark.e2e
def test_describe_scene_e2e():
    """Scene description with mock detections (no GPU)."""
    out = scene.describe_scene([], face_count=0)
    assert out == "No notable objects"
    out = scene.describe_scene([{"cls": 0, "conf": 0.9}], face_count=1)
    assert "person" in out
    assert "face" in out.lower()


@pytest.mark.e2e
def test_camera_open_invalid_index():
    """Opening invalid camera index returns None or unopened cap."""
    cap = camera.open_camera(index=999, width=640, height=480, fps=15)
    if cap is not None:
        assert not cap.isOpened()
        cap.release()


@pytest.mark.e2e
def test_yolo_engine_missing_returns_none():
    """Missing YOLOE engine path returns None."""
    model = detector_yolo.load_yolo_engine("/nonexistent/yoloe26n.engine")
    assert model is None
