"""Unit tests for vision scene description (basic + enriched)."""

import pytest
from vision.scene import describe_scene, describe_scene_enriched


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


# ── Enriched scene description ────────────────────────────────────────


@pytest.mark.unit
def test_enriched_empty():
    out = describe_scene_enriched([])
    assert out == "No notable objects"


@pytest.mark.unit
def test_enriched_with_detections():
    dets = [
        {"cls": 0, "conf": 0.9},
        {"cls": 41, "conf": 0.8},  # cup
    ]
    out = describe_scene_enriched(dets, face_count=1)
    assert "person" in out or "cup" in out
    assert "face" in out.lower()


@pytest.mark.unit
def test_enriched_with_vitals():
    from vision.vitals import VitalsResult

    vitals = VitalsResult(fatigue_level="mild", posture_label="fair")
    out = describe_scene_enriched([], vitals=vitals)
    assert "fatigue" in out.lower() or "mild" in out.lower()
    assert "posture" in out.lower()


@pytest.mark.unit
def test_enriched_with_threat():
    from vision.threat import ThreatAssessment

    threat = ThreatAssessment(level=3, label="low")
    out = describe_scene_enriched([], threat=threat)
    assert "Threat" in out
    assert "3/10" in out


@pytest.mark.unit
def test_enriched_full():
    from vision.threat import ThreatAssessment
    from vision.vitals import VitalsResult

    dets = [{"cls": 0, "conf": 0.9}]
    vitals = VitalsResult(fatigue_level="moderate", posture_label="good", heart_rate_bpm=72.0, heart_rate_confidence=0.8)
    threat = ThreatAssessment(level=1, label="clear")
    out = describe_scene_enriched(dets, face_count=1, vitals=vitals, threat=threat)
    assert "person" in out.lower()
    assert "fatigue" in out.lower() or "moderate" in out.lower()
    assert "Threat" in out
