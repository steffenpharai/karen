"""Unit tests for vision/ego_motion.py — camera ego-motion estimation."""


import numpy as np


class TestEstimateEgoMotion:
    """Tests for the ego-motion estimator."""

    def test_no_points_returns_static(self):
        """No matched points → static result."""
        from vision.ego_motion import estimate_ego_motion

        result = estimate_ego_motion(None, None)
        assert not result.is_moving
        assert result.motion_type == "static"
        assert result.ego_dx == 0.0
        assert result.ego_dy == 0.0

    def test_too_few_points_returns_static(self):
        """Fewer than min_points → static."""
        from vision.ego_motion import estimate_ego_motion

        prev = np.random.rand(5, 2).astype(np.float32)
        curr = prev + 0.1
        result = estimate_ego_motion(prev, curr, min_points=10)
        assert result.motion_type == "static"

    def test_small_motion_is_static(self):
        """Motion below threshold → static."""
        from vision.ego_motion import estimate_ego_motion

        np.random.seed(42)
        prev = np.random.rand(50, 2).astype(np.float64) * 320
        curr = prev + 0.5  # tiny motion
        result = estimate_ego_motion(prev, curr, motion_threshold=2.0)
        assert result.motion_type == "static"
        assert not result.is_moving

    def test_large_horizontal_motion_is_panning(self):
        """Large horizontal-dominant motion → panning."""
        from vision.ego_motion import estimate_ego_motion

        np.random.seed(42)
        prev = np.random.rand(80, 2).astype(np.float64) * 200 + 60
        # Shift everything right by 10 pixels (simulates panning)
        curr = prev.copy()
        curr[:, 0] += 10.0
        result = estimate_ego_motion(prev, curr, motion_threshold=1.0)
        assert result.is_moving
        assert result.ego_dx > 5.0
        assert abs(result.ego_dy) < abs(result.ego_dx)

    def test_large_uniform_motion_detected(self):
        """All points moving uniformly → camera is moving."""
        from vision.ego_motion import estimate_ego_motion

        np.random.seed(42)
        prev = np.random.rand(100, 2).astype(np.float64) * 300 + 10
        curr = prev + np.array([5.0, 3.0])  # uniform shift
        result = estimate_ego_motion(prev, curr, motion_threshold=1.0)
        assert result.is_moving
        assert result.num_inliers > 50

    def test_inlier_ratio_reasonable(self):
        """With uniform motion + no outliers, inlier ratio should be high."""
        from vision.ego_motion import estimate_ego_motion

        np.random.seed(42)
        prev = np.random.rand(100, 2).astype(np.float64) * 300 + 10
        curr = prev + np.array([4.0, 2.0])
        result = estimate_ego_motion(prev, curr, motion_threshold=1.0)
        assert result.inlier_ratio > 0.5


class TestCompensateEgoMotion:
    """Tests for ego-motion subtraction."""

    def test_compensation_subtracts_ego(self):
        from vision.ego_motion import EgoMotionResult, compensate_ego_motion

        ego = EgoMotionResult(ego_dx=5.0, ego_dy=3.0, is_moving=True)
        flows = [(8.0, 6.0), (5.0, 3.0), None]
        compensated = compensate_ego_motion(flows, ego)

        assert compensated[0] == (3.0, 3.0)  # 8-5, 6-3
        assert compensated[1] == (0.0, 0.0)  # 5-5, 3-3 → stationary object
        assert compensated[2] is None

    def test_static_ego_no_change(self):
        from vision.ego_motion import EgoMotionResult, compensate_ego_motion

        ego = EgoMotionResult(ego_dx=0.0, ego_dy=0.0)
        flows = [(3.0, 4.0)]
        compensated = compensate_ego_motion(flows, ego)
        assert compensated[0] == (3.0, 4.0)


class TestFlowToVelocity:
    """Tests for pixel flow → m/s conversion."""

    def test_no_depth_returns_none(self):
        from vision.ego_motion import flow_to_velocity_mps

        assert flow_to_velocity_mps(5.0, 3.0, None) is None
        assert flow_to_velocity_mps(5.0, 3.0, 0.0) is None

    def test_positive_velocity(self):
        from vision.ego_motion import flow_to_velocity_mps

        result = flow_to_velocity_mps(
            flow_dx=5.0, flow_dy=0.0,
            depth_relative=0.5,
            fps=30.0,
            frame_width=320,
            hfov_deg=60.0,
        )
        assert result is not None
        vx, vy, speed = result
        assert speed > 0
        assert abs(vy) < abs(vx)  # horizontal motion

    def test_speed_increases_with_depth(self):
        """Same pixel flow at greater depth → faster real-world speed."""
        from vision.ego_motion import flow_to_velocity_mps

        near = flow_to_velocity_mps(5.0, 0.0, 0.2, fps=30)
        far = flow_to_velocity_mps(5.0, 0.0, 0.8, fps=30)
        assert near is not None and far is not None
        assert far[2] > near[2]

    def test_speed_increases_with_fps(self):
        """Higher FPS with same pixel flow → faster velocity."""
        from vision.ego_motion import flow_to_velocity_mps

        slow_fps = flow_to_velocity_mps(5.0, 0.0, 0.5, fps=10)
        fast_fps = flow_to_velocity_mps(5.0, 0.0, 0.5, fps=30)
        assert slow_fps is not None and fast_fps is not None
        assert fast_fps[2] > slow_fps[2]


class TestClassifyMotion:
    """Tests for motion classification."""

    def test_static(self):
        from vision.ego_motion import _classify_motion

        assert _classify_motion(0.5, 0.3, 0.8) == "static"

    def test_panning(self):
        from vision.ego_motion import _classify_motion

        assert _classify_motion(10.0, 1.0, 10.0) == "panning"

    def test_tilting(self):
        from vision.ego_motion import _classify_motion

        assert _classify_motion(1.0, 10.0, 10.0) == "tilting"

    def test_walking(self):
        from vision.ego_motion import _classify_motion

        assert _classify_motion(4.0, 4.0, 8.0) == "walking"
