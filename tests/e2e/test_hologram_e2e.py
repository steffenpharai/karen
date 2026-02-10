"""E2E tests for hologram data generation (point cloud + tracked objects)."""

import numpy as np
import pytest
from vision.depth import depth_at_boxes, generate_point_cloud
from vision.tracker import ByteTrackLite


@pytest.mark.e2e
def test_point_cloud_from_synthetic_depth():
    """Point cloud generation from a synthetic depth map."""
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    depth = np.random.rand(240, 320).astype(np.float32) * 0.8 + 0.1
    points = generate_point_cloud(frame, depth, sample_step=10, max_points=1000)
    assert len(points) > 0
    assert len(points) <= 1000
    # Check structure
    p = points[0]
    assert "x" in p and "y" in p and "z" in p
    assert "r" in p and "g" in p and "b" in p
    assert isinstance(p["x"], float)
    assert 0 <= p["r"] <= 255


@pytest.mark.e2e
def test_tracker_with_depth_integration():
    """Tracker + depth_at_boxes produces tracked objects with depth."""
    tracker = ByteTrackLite(min_hits=1)
    dets = [
        {"xyxy": [10, 10, 100, 100], "conf": 0.9, "cls": 0, "class_name": "person"},
        {"xyxy": [200, 200, 280, 280], "conf": 0.8, "cls": 56, "class_name": "chair"},
    ]
    tracked = tracker.update(dets)
    assert len(tracked) == 2

    # Depth map
    depth_map = np.random.rand(300, 300).astype(np.float32)
    depths = depth_at_boxes(depth_map, dets)
    assert len(depths) == 2
    assert all(d is not None for d in depths)


@pytest.mark.e2e
def test_hologram_pipeline_integration():
    """Full pipeline: detect -> track -> depth -> point cloud."""
    # Simulate detections
    dets = [{"xyxy": [50, 50, 150, 150], "conf": 0.9, "cls": 0, "class_name": "person"}]

    # Track
    tracker = ByteTrackLite(min_hits=1)
    tracked = tracker.update(dets)
    assert len(tracked) == 1

    # Depth
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    depth_map = np.random.rand(200, 200).astype(np.float32) * 0.5 + 0.2
    depths = depth_at_boxes(depth_map, dets)
    tracked[0].depth = depths[0]

    # Point cloud
    points = generate_point_cloud(frame, depth_map, sample_step=20)
    assert len(points) > 0

    # Serializable for WebSocket
    payload = {
        "point_cloud": points[:100],
        "tracked_objects": [
            {
                "track_id": t.track_id,
                "xyxy": t.xyxy,
                "class_name": t.class_name,
                "depth": t.depth,
            }
            for t in tracked
        ],
    }
    assert len(payload["tracked_objects"]) == 1
    assert payload["tracked_objects"][0]["depth"] is not None
