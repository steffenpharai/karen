"""Unit tests for vision/trajectory.py — trajectory prediction & collision detection."""

from dataclasses import dataclass, field

import pytest


@dataclass
class MockTrackedObject:
    """Minimal mock of TrackedObject for trajectory tests."""

    track_id: int = 1
    xyxy: list = field(default_factory=lambda: [100, 100, 200, 200])
    cls: int = 0
    class_name: str = "person"
    conf: float = 0.9
    velocity: list = field(default_factory=lambda: [0.0, 0.0])
    depth: float | None = None


class TestTrajectoryPredictor:
    """Tests for TrajectoryPredictor."""

    def test_stationary_object(self):
        """Object with no velocity → stationary behaviour, no alerts."""
        from vision.trajectory import TrajectoryPredictor

        predictor = TrajectoryPredictor()
        obj = MockTrackedObject(velocity=[0.0, 0.0])
        trajectories, alerts = predictor.predict_all(
            [obj], frame_size=(320, 240), fps=30,
        )
        assert len(trajectories) == 1
        assert trajectories[0].behaviour == "stationary"
        assert len(alerts) == 0

    def test_approaching_object_generates_alert(self):
        """Object approaching with depth and velocity → approaching or moving."""
        from vision.trajectory import TrajectoryPredictor

        predictor = TrajectoryPredictor(
            prediction_horizon_sec=3.0,
            collision_zone_m=2.0,
        )
        # Object at left edge, moving toward camera center (right)
        obj = MockTrackedObject(
            track_id=1,
            xyxy=[20, 100, 60, 200],
            velocity=[100.0, 0.0],  # moving right toward center
            depth=0.3,  # ~3m away
        )
        trajectories, alerts = predictor.predict_all(
            [obj],
            velocity_mps_list=[(1.5, 0.0, 1.5)],
            frame_size=(320, 240),
            fps=30,
        )
        assert len(trajectories) == 1
        traj = trajectories[0]
        assert traj.behaviour in ("approaching", "moving", "crossing")
        assert traj.depth_m is not None
        assert traj.depth_m == pytest.approx(3.0)

    def test_waypoints_generated(self):
        """Moving object should have waypoints projected forward."""
        from vision.trajectory import TrajectoryPredictor

        predictor = TrajectoryPredictor(
            prediction_horizon_sec=2.0,
            prediction_steps=4,
        )
        obj = MockTrackedObject(velocity=[50.0, 30.0])
        trajectories, _ = predictor.predict_all(
            [obj], frame_size=(320, 240), fps=30,
        )
        assert len(trajectories[0].waypoints) == 4
        # Each waypoint should be (cx, cy, t_sec)
        for wp in trajectories[0].waypoints:
            assert len(wp) == 3
            assert wp[2] > 0  # positive time

    def test_multiple_objects(self):
        """Should handle multiple tracked objects."""
        from vision.trajectory import TrajectoryPredictor

        predictor = TrajectoryPredictor()
        objs = [
            MockTrackedObject(track_id=1, velocity=[50, 0]),
            MockTrackedObject(track_id=2, velocity=[0, 50]),
            MockTrackedObject(track_id=3, velocity=[0, 0]),
        ]
        trajectories, alerts = predictor.predict_all(
            objs, frame_size=(320, 240), fps=30,
        )
        assert len(trajectories) == 3

    def test_reset_clears_state(self):
        """Reset should clear velocity history."""
        from vision.trajectory import TrajectoryPredictor

        predictor = TrajectoryPredictor()
        obj = MockTrackedObject(velocity=[50, 30])
        predictor.predict_all([obj], frame_size=(320, 240), fps=30)
        assert len(predictor._prev_velocities) > 0
        predictor.reset()
        assert len(predictor._prev_velocities) == 0


class TestCollisionAlert:
    """Tests for collision alert generation."""

    def test_build_alert_critical(self):
        from vision.trajectory import _build_alert

        alert = _build_alert(
            track_id=1,
            class_name="person",
            speed_mps=2.0,
            distance_m=1.5,
            ttc=0.8,
            direction="ahead",
        )
        assert alert is not None
        assert alert.severity == "critical"
        assert "person" in alert.message
        assert "Sir" in alert.message

    def test_build_alert_warning(self):
        from vision.trajectory import _build_alert

        alert = _build_alert(
            track_id=1,
            class_name="bicycle",
            speed_mps=3.0,
            distance_m=3.0,
            ttc=1.5,
            direction="left",
        )
        assert alert is not None
        assert alert.severity == "warning"
        assert "bicycle" in alert.message

    def test_no_alert_for_distant_slow(self):
        """Far away, slow object → no alert."""
        from vision.trajectory import _build_alert

        alert = _build_alert(
            track_id=1,
            class_name="person",
            speed_mps=0.5,
            distance_m=10.0,
            ttc=20.0,
            direction="ahead",
        )
        assert alert is None

    def test_no_alert_zero_ttc(self):
        from vision.trajectory import _build_alert

        assert _build_alert(1, "obj", 1.0, 1.0, 0.0, "ahead") is None


class TestClassifyBehaviour:
    """Tests for object behaviour classification."""

    def test_stationary(self):
        from vision.trajectory import _classify_behaviour

        assert _classify_behaviour(0, 0, 160, 120, 320, 240, 5) == "stationary"

    def test_approaching(self):
        from vision.trajectory import _classify_behaviour

        # Object at left, moving right toward center
        result = _classify_behaviour(50, 0, 50, 120, 320, 240, 50)
        assert result == "approaching"

    def test_receding(self):
        from vision.trajectory import _classify_behaviour

        # Object at center-right, moving further right (away from center)
        result = _classify_behaviour(50, 0, 250, 120, 320, 240, 50)
        assert result == "receding"


class TestFormatTrajectorySummary:
    """Tests for trajectory text summary formatting."""

    def test_empty_input(self):
        from vision.trajectory import format_trajectory_summary

        assert format_trajectory_summary([], []) == ""

    def test_ego_motion_included(self):
        from vision.trajectory import format_trajectory_summary

        result = format_trajectory_summary([], [], ego_motion_type="walking")
        assert "walking" in result

    def test_alerts_included(self):
        from vision.trajectory import CollisionAlert, format_trajectory_summary

        alert = CollisionAlert(
            message="Person from left", severity="warning",
        )
        result = format_trajectory_summary([], [alert])
        assert "WARNING" in result
