"""Unit tests for vision scene description."""

import pytest
from vision.scene import describe_scene


@pytest.mark.unit
def test_describe_scene_empty():
    assert describe_scene([]) == "No notable objects"
    assert describe_scene([], face_count=0) == "No notable objects"


@pytest.mark.unit
def test_describe_scene_yolo_only():
    # cls 0 = person
    dets = [
        {"cls": 0, "conf": 0.9},
        {"cls": 0, "conf": 0.8},
        {"cls": 56, "conf": 0.7},
    ]  # 56 = chair
    out = describe_scene(dets)
    assert "person" in out
    assert "chair" in out
    assert "2" in out or "1" in out  # counts


@pytest.mark.unit
def test_describe_scene_faces():
    out = describe_scene([], face_count=2)
    assert "face" in out.lower()
    assert "2" in out
