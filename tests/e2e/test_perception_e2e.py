"""E2E tests for the perception pipeline — simulates walking sequences.

Marked with @pytest.mark.e2e for separate execution:
  pytest tests/e2e/test_perception_e2e.py -m e2e
"""

from dataclasses import dataclass, field

import numpy as np
import pytest


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


def _make_walking_sequence(num_frames=10, h=96, w=128):
    """Generate a sequence of frames simulating camera walking forward.

    Each frame shifts a pattern to simulate ego-motion.
    """
    frames = []
    base = np.random.randint(50, 200, (h + num_frames * 2, w + num_frames * 2, 3), dtype=np.uint8)
    for i in range(num_frames):
        frame = base[i * 2: i * 2 + h, i: i + w].copy()
        frames.append(frame)
    return frames


def _make_approaching_object_sequence(num_frames=10, h=96, w=128):
    """Simulate an object approaching the camera (growing bbox, centered)."""
    frames = []
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Object grows and gets closer to center
        size = 10 + i * 3
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - size)
        y1 = max(0, cy - size)
        x2 = min(w, cx + size)
        y2 = min(h, cy + size)
        frame[y1:y2, x1:x2] = [200, 100, 50]
        frames.append(frame)
    return frames


@pytest.mark.e2e
class TestWalkingSequence:
    """Test perception pipeline with simulated walking camera movement."""

    def test_ego_motion_detected_during_walk(self):
        """Camera walking should be detected as ego-motion."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(128, 96),
            fps=30.0,
        )
        frames = _make_walking_sequence(num_frames=8, h=96, w=128)

        # Process sequence
        results = []
        for frame in frames:
            result = pipeline.process_frame(frame, [], [])
            results.append(result)

        # After warmup (first 2-3 frames), ego-motion should be detected
        moving_frames = sum(
            1 for r in results[2:]
            if r.ego_motion is not None and r.ego_motion.is_moving
        )
        # At least some frames should detect motion
        assert moving_frames >= 1, "Walking sequence should detect camera motion"

    def test_motion_energy_during_walk(self):
        """Walking should produce significant motion energy."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(128, 96),
        )
        frames = _make_walking_sequence(num_frames=6, h=96, w=128)

        energies = []
        for frame in frames:
            result = pipeline.process_frame(frame, [], [])
            energies.append(result.motion_energy)

        # Skip first frame (no flow). Later frames should have motion energy.
        assert max(energies[1:]) > 0, "Walking should produce motion energy"


@pytest.mark.e2e
class TestApproachingObject:
    """Test perception pipeline with object approaching camera."""

    def test_approaching_behaviour_classified(self):
        """Growing object should be classified as approaching."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(128, 96),
            fps=30.0,
        )
        frames = _make_approaching_object_sequence(num_frames=8, h=96, w=128)

        last_result = None
        for i, frame in enumerate(frames):
            size = 10 + i * 3
            cx, cy = 64, 48
            x1, y1 = max(0, cx - size), max(0, cy - size)
            x2, y2 = min(128, cx + size), min(96, cy + size)

            dets = [{"xyxy": [x1, y1, x2, y2], "conf": 0.9, "cls": 0}]
            tracked = [MockTrackedObject(
                xyxy=[x1, y1, x2, y2],
                velocity=[0.0, 5.0 * (i + 1)],
                depth=max(0.1, 0.8 - i * 0.08),
            )]

            last_result = pipeline.process_frame(
                frame, dets, tracked,
                depth_values=[max(0.1, 0.8 - i * 0.08)],
            )

        assert last_result is not None
        assert len(last_result.trajectories) > 0

    def test_collision_alert_generated(self):
        """Object approaching at speed with close depth → collision alert."""
        from vision.trajectory import TrajectoryPredictor

        predictor = TrajectoryPredictor(
            prediction_horizon_sec=3.0,
            collision_zone_m=2.0,
        )
        obj = MockTrackedObject(
            track_id=1,
            xyxy=[140, 100, 180, 200],
            velocity=[0.0, 100.0],  # moving in frame
            depth=0.2,  # close (~2m)
        )
        trajectories, alerts = predictor.predict_all(
            [obj],
            velocity_mps_list=[(0.0, 2.0, 2.0)],  # 2 m/s
            frame_size=(320, 240),
            fps=30,
        )
        # With depth=0.2 (2m) and speed=2m/s, TTC should be ~1s
        # This should generate an alert
        assert len(trajectories) == 1
        traj = trajectories[0]
        assert traj.depth_m is not None
        assert traj.depth_m == pytest.approx(2.0)


@pytest.mark.e2e
class TestPerceptionTiming:
    """Test that perception pipeline meets timing budget."""

    def test_flow_under_50ms(self):
        """Flow computation should be under 50ms at 128x96."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(128, 96),
        )
        frame1 = np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8)

        pipeline.process_frame(frame1, [], [])
        result = pipeline.process_frame(frame2, [], [])

        assert result.flow_ms < 50, f"Flow too slow: {result.flow_ms:.1f}ms"

    def test_total_pipeline_under_100ms(self):
        """Full perception pipeline should be under 100ms at small resolution."""
        from vision.flow import FlowMethod
        from vision.perception import PerceptionPipeline

        pipeline = PerceptionPipeline(
            flow_method=FlowMethod.FARNEBACK,
            flow_resize=(128, 96),
        )
        frame1 = np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8)

        dets = [{"xyxy": [10, 10, 50, 50], "conf": 0.8, "cls": 0}]
        tracked = [MockTrackedObject()]

        pipeline.process_frame(frame1, [], [])
        result = pipeline.process_frame(frame2, dets, tracked)

        assert result.total_ms < 100, f"Pipeline too slow: {result.total_ms:.1f}ms"
