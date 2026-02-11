"""Unit tests for vision/flow.py — optical flow estimation."""

import numpy as np


class TestOpticalFlowEstimator:
    """Tests for the stateful optical flow estimator."""

    def test_first_frame_returns_empty(self):
        """First frame has no previous frame → empty FlowResult."""
        from vision.flow import FlowMethod, OpticalFlowEstimator

        estimator = OpticalFlowEstimator(
            method=FlowMethod.FARNEBACK, resize=(64, 48)
        )
        frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        result = estimator.compute(frame)
        assert result.flow is None
        assert result.mean_magnitude == 0.0

    def test_second_frame_produces_flow(self):
        """Two frames should produce a valid dense flow field."""
        from vision.flow import FlowMethod, OpticalFlowEstimator

        estimator = OpticalFlowEstimator(
            method=FlowMethod.FARNEBACK, resize=(64, 48)
        )
        frame1 = np.zeros((48, 64, 3), dtype=np.uint8)
        frame2 = np.zeros((48, 64, 3), dtype=np.uint8)
        # Add a moving rectangle
        frame1[10:20, 10:20] = 255
        frame2[10:20, 15:25] = 255  # shifted right by 5 pixels

        estimator.compute(frame1)
        result = estimator.compute(frame2)

        assert result.flow is not None
        assert result.flow.shape == (48, 64, 2)
        assert result.magnitude is not None
        assert result.max_magnitude >= 0

    def test_dis_method(self):
        """DIS optical flow should also produce valid results."""
        from vision.flow import FlowMethod, OpticalFlowEstimator

        estimator = OpticalFlowEstimator(
            method=FlowMethod.DIS, resize=(64, 48)
        )
        frame1 = np.random.randint(50, 200, (48, 64, 3), dtype=np.uint8)
        frame2 = np.random.randint(50, 200, (48, 64, 3), dtype=np.uint8)

        estimator.compute(frame1)
        result = estimator.compute(frame2)

        assert result.flow is not None
        assert result.flow.shape == (48, 64, 2)

    def test_sparse_flow_points(self):
        """Sparse flow should produce matched keypoints."""
        from vision.flow import FlowMethod, OpticalFlowEstimator

        estimator = OpticalFlowEstimator(
            method=FlowMethod.FARNEBACK, resize=(128, 96),
            sparse_max_corners=50,
        )
        # Create textured frames (corners need texture to detect)
        frame1 = np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8)
        frame2 = frame1.copy()

        estimator.compute(frame1)
        result = estimator.compute(frame2)

        # With textured random images, we should get matched points
        if result.prev_points is not None:
            assert len(result.prev_points) > 0
            assert len(result.curr_points) == len(result.prev_points)

    def test_reset_clears_state(self):
        """After reset, next compute should behave like first frame."""
        from vision.flow import FlowMethod, OpticalFlowEstimator

        estimator = OpticalFlowEstimator(
            method=FlowMethod.FARNEBACK, resize=(64, 48)
        )
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        estimator.compute(frame)
        estimator.reset()
        result = estimator.compute(frame)
        assert result.flow is None  # treated as first frame again

    def test_compute_time_tracked(self):
        """Compute time should be recorded."""
        from vision.flow import FlowMethod, OpticalFlowEstimator

        estimator = OpticalFlowEstimator(
            method=FlowMethod.FARNEBACK, resize=(64, 48)
        )
        frame1 = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        estimator.compute(frame1)
        result = estimator.compute(frame2)
        assert result.compute_time_ms >= 0


class TestFlowAtBoxes:
    """Tests for per-box flow extraction."""

    def test_no_flow_returns_nones(self):
        from vision.flow import flow_at_boxes

        dets = [{"xyxy": [10, 10, 20, 20]}]
        result = flow_at_boxes(None, dets)
        assert result == [None]

    def test_valid_flow_returns_vectors(self):
        from vision.flow import flow_at_boxes

        # Create a uniform flow field (everything moving right by 3 pixels)
        flow = np.zeros((48, 64, 2), dtype=np.float32)
        flow[..., 0] = 3.0  # dx = 3

        dets = [{"xyxy": [10, 10, 30, 30]}]
        result = flow_at_boxes(flow, dets, frame_shape=(48, 64))

        assert result[0] is not None
        dx, dy = result[0]
        assert abs(dx - 3.0) < 0.5  # should be ~3.0

    def test_scaling_for_different_resolutions(self):
        """Flow at different resolution than frame should scale coords."""
        from vision.flow import flow_at_boxes

        flow = np.ones((24, 32, 2), dtype=np.float32)  # half resolution
        dets = [{"xyxy": [0, 0, 64, 48]}]  # full resolution coords
        result = flow_at_boxes(flow, dets, frame_shape=(48, 64))
        assert result[0] is not None


class TestMotionEnergy:
    """Tests for compute_motion_energy."""

    def test_static_scene_zero_energy(self):
        from vision.flow import compute_motion_energy

        flow = np.zeros((48, 64, 2), dtype=np.float32)
        assert compute_motion_energy(flow) == 0.0

    def test_moving_scene_nonzero_energy(self):
        from vision.flow import compute_motion_energy

        flow = np.ones((48, 64, 2), dtype=np.float32) * 5.0
        energy = compute_motion_energy(flow, threshold=1.0)
        assert energy > 0.5

    def test_none_flow_returns_zero(self):
        from vision.flow import compute_motion_energy

        assert compute_motion_energy(None) == 0.0
