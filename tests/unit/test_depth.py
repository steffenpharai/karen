"""Unit tests for vision/depth.py: depth estimation, depth_at_boxes, point cloud."""


import numpy as np
import pytest
from vision.depth import depth_at_boxes, generate_point_cloud, load_depth_model

# ── depth_at_boxes ────────────────────────────────────────────────────


@pytest.mark.unit
class TestDepthAtBoxes:
    def test_none_depth_map(self):
        dets = [{"xyxy": [0, 0, 10, 10]}]
        result = depth_at_boxes(None, dets)
        assert result == [None]

    def test_empty_detections(self):
        depth_map = np.ones((100, 100), dtype=np.float32) * 0.5
        result = depth_at_boxes(depth_map, [])
        assert result == []

    def test_single_box(self):
        depth_map = np.ones((100, 100), dtype=np.float32) * 0.7
        dets = [{"xyxy": [10, 10, 50, 50]}]
        result = depth_at_boxes(depth_map, dets)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.7)

    def test_multiple_boxes(self):
        depth_map = np.zeros((100, 100), dtype=np.float32)
        depth_map[10:50, 10:50] = 0.3
        depth_map[60:90, 60:90] = 0.8
        dets = [
            {"xyxy": [10, 10, 50, 50]},
            {"xyxy": [60, 60, 90, 90]},
        ]
        result = depth_at_boxes(depth_map, dets)
        assert len(result) == 2
        assert result[0] == pytest.approx(0.3, abs=0.1)
        assert result[1] == pytest.approx(0.8, abs=0.1)

    def test_invalid_box(self):
        depth_map = np.ones((100, 100), dtype=np.float32)
        dets = [{"xyxy": None}]
        result = depth_at_boxes(depth_map, dets)
        assert result == [None]

    def test_zero_area_box(self):
        depth_map = np.ones((100, 100), dtype=np.float32)
        dets = [{"xyxy": [50, 50, 50, 50]}]
        result = depth_at_boxes(depth_map, dets)
        assert result == [None]

    def test_out_of_bounds_box(self):
        depth_map = np.ones((100, 100), dtype=np.float32) * 0.5
        dets = [{"xyxy": [0, 0, 200, 200]}]
        result = depth_at_boxes(depth_map, dets)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.5)


# ── generate_point_cloud ──────────────────────────────────────────────


@pytest.mark.unit
class TestGeneratePointCloud:
    def test_none_inputs(self):
        assert generate_point_cloud(None, None) == []
        assert generate_point_cloud(np.zeros((100, 100, 3)), None) == []

    def test_basic_generation(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        depth = np.random.rand(100, 100).astype(np.float32) * 0.5 + 0.1
        points = generate_point_cloud(frame, depth, sample_step=20)
        assert len(points) > 0
        assert "x" in points[0]
        assert "y" in points[0]
        assert "z" in points[0]
        assert "r" in points[0]
        assert 0 <= points[0]["r"] <= 255

    def test_max_points(self):
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        depth = np.ones((200, 200), dtype=np.float32) * 0.5
        points = generate_point_cloud(frame, depth, sample_step=2, max_points=100)
        assert len(points) <= 100

    def test_near_zero_depth_skipped(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        depth = np.zeros((100, 100), dtype=np.float32)  # all near-zero
        points = generate_point_cloud(frame, depth, sample_step=10)
        assert len(points) == 0


# ── load_depth_model ──────────────────────────────────────────────────


@pytest.mark.unit
def test_load_nonexistent_engine():
    result = load_depth_model("/nonexistent/depth.engine")
    # Should return None (no engine, no fallback without torch hub)
    assert result is None or isinstance(result, dict)
