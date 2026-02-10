"""Unit tests for vision/threat.py: ThreatScorer heuristics."""

import time

import pytest
from vision.threat import ThreatAssessment, ThreatScorer
from vision.tracker import TrackedObject


def _make_track(
    track_id: int = 1,
    cls: int = 0,
    class_name: str = "person",
    xyxy: list | None = None,
    velocity: list | None = None,
    depth: float | None = None,
    frames_seen: int = 5,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        xyxy=xyxy or [100, 100, 200, 200],
        cls=cls,
        class_name=class_name,
        conf=0.9,
        velocity=velocity or [0.0, 0.0],
        frames_seen=frames_seen,
        age=0,
        last_seen=time.monotonic(),
        depth=depth,
    )


@pytest.mark.unit
class TestThreatScorer:
    def test_empty_scene_clear(self):
        scorer = ThreatScorer()
        result = scorer.score_scene([], None, None)
        assert isinstance(result, ThreatAssessment)
        assert result.level == 0
        assert result.label == "clear"
        assert result.recommendation == ""

    def test_peaceful_scene(self):
        scorer = ThreatScorer()
        tracks = [_make_track(velocity=[0, 0])]
        result = scorer.score_scene(tracks, None, None)
        assert result.level <= 2
        assert result.label == "clear"

    def test_rapid_approach(self):
        scorer = ThreatScorer()
        tracks = [_make_track(velocity=[100.0, 100.0])]
        result = scorer.score_scene(tracks, None, None)
        assert result.level >= 1
        assert len(result.alerts) > 0

    def test_close_proximity(self):
        scorer = ThreatScorer()
        tracks = [_make_track(depth=0.05)]  # very close
        result = scorer.score_scene(tracks, None, None)
        assert result.level >= 1

    def test_weapon_detection(self):
        scorer = ThreatScorer()
        tracks = [
            _make_track(track_id=1, class_name="person"),
            _make_track(track_id=2, cls=43, class_name="knife"),
        ]
        result = scorer.score_scene(tracks, None, None)
        assert result.level >= 1
        assert any("weapon" in a.lower() or "knife" in a.lower() for a in result.alerts)

    def test_crowd_surge(self):
        scorer = ThreatScorer()
        # First frame: empty
        scorer.score_scene([], None, None)
        # Second frame: 3+ new objects at once
        tracks = [
            _make_track(track_id=i, class_name="person")
            for i in range(5)
        ]
        result = scorer.score_scene(tracks, None, None)
        assert result.level >= 1

    def test_fast_vehicle(self):
        scorer = ThreatScorer()
        tracks = [_make_track(class_name="car", velocity=[200.0, 0.0])]
        result = scorer.score_scene(tracks, None, None)
        assert result.level >= 2

    def test_smoothing(self):
        scorer = ThreatScorer(smoothing_frames=3)
        # Spike then calm
        tracks_danger = [_make_track(velocity=[200.0, 200.0], depth=0.05)]
        scorer.score_scene(tracks_danger, None, None)
        scorer.score_scene([], None, None)
        result = scorer.score_scene([], None, None)
        # Should be smoothed down
        assert result.level < 10

    def test_reset(self):
        scorer = ThreatScorer()
        scorer.score_scene([_make_track()], None, None)
        scorer.reset()
        assert scorer._level_history == []
        assert scorer._prev_person_count == 0

    def test_level_labels(self):
        assert ThreatScorer._level_to_label(0) == "clear"
        assert ThreatScorer._level_to_label(3) == "low"
        assert ThreatScorer._level_to_label(5) == "moderate"
        assert ThreatScorer._level_to_label(7) == "elevated"
        assert ThreatScorer._level_to_label(9) == "critical"

    def test_recommendations(self):
        assert ThreatScorer._recommend(0, []) == ""
        assert "sir" in ThreatScorer._recommend(5, ["test"]).lower()
        assert "evasive" in ThreatScorer._recommend(9, ["danger"]).lower()

    def test_vitals_factor(self):
        """Fatigue + anomalous scene should increase threat."""
        scorer = ThreatScorer()
        from vision.vitals import VitalsResult

        vitals = VitalsResult(fatigue_level="severe")
        tracks = [_make_track(velocity=[60.0, 60.0])]  # moving
        result = scorer.score_scene(tracks, vitals, None)
        # Should be higher than without fatigue
        scorer2 = ThreatScorer()
        result2 = scorer2.score_scene(tracks, None, None)
        assert result.level >= result2.level
