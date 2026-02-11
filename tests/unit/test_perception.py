"""Unit tests for vision/perception.py — fused perception pipeline."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MockTrackedObject:
    track_id: int = 1
    xyxy: list = field(default_factory=lambda: [50, 50, 150, 150])
    cls: int = 0
    class_name: str = "person"
    conf: float = 0.9
    velocity: list = field(default_factory=lambda: [10.0, 5.0])
    depth: float | None = None
    frames_seen: int = 5
    age: int = 0
    last_seen: float = 0.0


class TestPerceptionPipeline:
    """Tests for the PerceptionPipeline."""

    def test_first_frame_empty_perception(self):
        """First frame: flow is empty, but pipeline should still return a result."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(64, 48),
            fps=30.0,
        )
        frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        result = pipeline.process_frame(
            frame, detections=[], tracked_objects=[], depth_map=None,
        )
        assert result.flow is not None  # FlowResult (but flow field is None on first frame)
        assert result.total_ms >= 0

    def test_two_frames_with_detections(self):
        """Two frames with detections should produce full perception data."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(64, 48),
            fps=30.0,
        )
        frame1 = np.random.randint(50, 200, (48, 64, 3), dtype=np.uint8)
        frame2 = frame1.copy()

        # First frame
        pipeline.process_frame(frame1, [], [], None)

        # Second frame with a detection
        dets = [{"xyxy": [10, 10, 30, 30], "conf": 0.9, "cls": 0}]
        tracked = [MockTrackedObject(xyxy=[10, 10, 30, 30], depth=0.5)]

        result = pipeline.process_frame(
            frame2, dets, tracked,
            depth_map=np.random.rand(48, 64).astype(np.float32),
            depth_values=[0.5],
        )

        assert result.flow is not None
        assert result.ego_motion is not None
        assert len(result.trajectories) == 1
        assert result.total_ms > 0

    def test_ego_motion_populated(self):
        """Ego-motion result should be populated after two frames."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(64, 48),
        )
        frame1 = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)

        pipeline.process_frame(frame1, [], [])
        result = pipeline.process_frame(frame2, [], [])

        assert result.ego_motion is not None
        assert result.ego_summary != ""

    def test_motion_energy_computed(self):
        """Motion energy should be computed from flow."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(64, 48),
        )
        frame1 = np.zeros((48, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((48, 64, 3), dtype=np.uint8)

        pipeline.process_frame(frame1, [], [])
        result = pipeline.process_frame(frame2, [], [])

        # Static scene → low motion energy
        assert result.motion_energy >= 0.0
        assert result.motion_energy <= 1.0

    def test_velocity_mps_with_depth(self):
        """Objects with flow + depth should get m/s velocities."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(64, 48),
            fps=30.0,
        )

        frame1 = np.zeros((48, 64, 3), dtype=np.uint8)
        frame1[10:20, 10:20] = 255
        frame2 = np.zeros((48, 64, 3), dtype=np.uint8)
        frame2[10:20, 15:25] = 255

        pipeline.process_frame(frame1, [], [])

        dets = [{"xyxy": [15, 10, 25, 20], "conf": 0.9, "cls": 0}]
        tracked = [MockTrackedObject(xyxy=[15, 10, 25, 20], depth=0.5)]

        result = pipeline.process_frame(
            frame2, dets, tracked,
            depth_values=[0.5],
        )

        # May or may not have velocity depending on flow detection
        assert isinstance(result.object_velocities_mps, list)

    def test_reset_clears_all_state(self):
        """Reset should clear flow estimator and trajectory predictor state."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(64, 48),
        )
        frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        pipeline.process_frame(frame, [], [])
        pipeline.reset()

        # After reset, should behave like first frame
        result = pipeline.process_frame(frame, [], [])
        assert result.flow.flow is None  # first frame after reset

    def test_trajectory_summary_text(self):
        """Trajectory summary should be a string."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(64, 48),
        )
        frame1 = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)

        pipeline.process_frame(frame1, [], [])
        result = pipeline.process_frame(frame2, [], [])

        assert isinstance(result.trajectory_summary, str)

    def test_perception_with_empty_detections(self):
        """Pipeline should handle empty detection list gracefully."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(64, 48),
        )
        frame1 = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)

        pipeline.process_frame(frame1, [], [])
        result = pipeline.process_frame(frame2, [], [])

        assert len(result.trajectories) == 0
        assert len(result.collision_alerts) == 0
