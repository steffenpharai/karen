"""Ego-motion estimation: camera rotation + translation from optical flow.

Decomposes the flow field into ego-motion (camera movement) and independent
object motion.  This is critical for walk-around mode — without ego-motion
compensation, the tracker would see every pixel shifting when the user turns
their head, causing false velocity readings on stationary objects.

Approach (Tesla FSD / SpaceX Dragon inspired):
  1. Sparse feature matching (Lucas-Kanade) from flow.py
  2. Estimate fundamental matrix (RANSAC) → inliers = static background
  3. Decompose fundamental → rotation + translation (up to scale)
  4. Median flow of inliers = ego-motion vector in pixel space
  5. Subtract ego-motion from per-object flow → true object velocity

The fundamental matrix approach is robust to moving objects (RANSAC
rejects them as outliers), making it ideal for walk-around scenarios.

Memory: ~0 extra — pure NumPy/OpenCV computation.
"""

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EgoMotionResult:
    """Estimated camera ego-motion between two consecutive frames."""

    # Pixel-space ego-motion vector (median flow of background)
    ego_dx: float = 0.0       # pixels/frame, horizontal
    ego_dy: float = 0.0       # pixels/frame, vertical

    # Camera rotation (Euler-ish, approximate from homography)
    yaw_deg: float = 0.0      # left-right turn
    pitch_deg: float = 0.0    # up-down tilt
    roll_deg: float = 0.0     # clockwise tilt

    # Translation direction (unit vector, scale-ambiguous for monocular)
    translation_dir: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Motion classification
    is_moving: bool = False    # True if significant camera motion detected
    motion_type: str = "static"  # static, walking, turning, panning

    # Quality
    inlier_ratio: float = 0.0  # fraction of flow points consistent with ego-motion
    num_inliers: int = 0
    num_points: int = 0


# ── Approximate camera intrinsics ─────────────────────────────────────

def _camera_matrix(w: int, h: int) -> np.ndarray:
    """Build approximate camera intrinsic matrix assuming ~60° HFOV."""
    fx = w / (2.0 * math.tan(math.radians(30)))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ── Core ego-motion estimator ─────────────────────────────────────────


def estimate_ego_motion(
    prev_points: np.ndarray | None,
    curr_points: np.ndarray | None,
    frame_size: tuple[int, int] = (320, 240),
    min_points: int = 15,
    motion_threshold: float = 1.5,
) -> EgoMotionResult:
    """Estimate camera ego-motion from sparse flow correspondences.

    Parameters
    ----------
    prev_points : Nx2 array of previous frame keypoints
    curr_points : Nx2 array of matched current frame keypoints
    frame_size : (W, H) of the flow computation frame
    min_points : minimum matched points to attempt estimation
    motion_threshold : mean flow below this → static (pixels/frame)

    Returns
    -------
    EgoMotionResult with ego-motion vector, rotation, and classification
    """
    result = EgoMotionResult()

    if (
        prev_points is None
        or curr_points is None
        or len(prev_points) < min_points
        or len(curr_points) < min_points
    ):
        return result

    n = min(len(prev_points), len(curr_points))
    prev_pts = prev_points[:n].reshape(-1, 2).astype(np.float64)
    curr_pts = curr_points[:n].reshape(-1, 2).astype(np.float64)
    result.num_points = n

    # Compute per-point flow
    flow_vecs = curr_pts - prev_pts
    mean_mag = float(np.mean(np.linalg.norm(flow_vecs, axis=1)))

    if mean_mag < motion_threshold:
        # Camera is essentially static
        result.motion_type = "static"
        result.inlier_ratio = 1.0
        result.num_inliers = n
        return result

    result.is_moving = True

    # ── Fundamental matrix via RANSAC ─────────────────────────────
    # Inliers correspond to static background (consistent with single
    # rigid motion = camera ego-motion).  Outliers = moving objects.
    try:
        F, mask = cv2.findFundamentalMat(
            prev_pts, curr_pts,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=2.0,
            confidence=0.99,
        )
    except cv2.error:
        # Degenerate configuration (e.g. all points collinear)
        F, mask = None, None

    if F is None or mask is None:
        # Fallback: use median flow as ego-motion estimate
        result.ego_dx = float(np.median(flow_vecs[:, 0]))
        result.ego_dy = float(np.median(flow_vecs[:, 1]))
        result.motion_type = _classify_motion(result.ego_dx, result.ego_dy, mean_mag)
        result.inlier_ratio = 1.0
        result.num_inliers = n
        return result

    inlier_mask = mask.ravel().astype(bool)
    result.num_inliers = int(np.sum(inlier_mask))
    result.inlier_ratio = result.num_inliers / n if n > 0 else 0.0

    # Ego-motion = median flow of inliers (background points)
    if result.num_inliers > 3:
        inlier_flows = flow_vecs[inlier_mask]
        result.ego_dx = float(np.median(inlier_flows[:, 0]))
        result.ego_dy = float(np.median(inlier_flows[:, 1]))
    else:
        result.ego_dx = float(np.median(flow_vecs[:, 0]))
        result.ego_dy = float(np.median(flow_vecs[:, 1]))

    # ── Estimate rotation from essential matrix ───────────────────
    w, h = frame_size
    K = _camera_matrix(w, h)

    try:
        E = K.T @ F @ K
        # Decompose essential matrix → R, t
        _, R, t, _ = cv2.recoverPose(E, prev_pts[inlier_mask], curr_pts[inlier_mask], K)

        # Extract approximate Euler angles from rotation matrix
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(-R[1, 2], R[1, 1])
            roll = 0.0

        result.yaw_deg = math.degrees(yaw)
        result.pitch_deg = math.degrees(pitch)
        result.roll_deg = math.degrees(roll)

        # Translation direction (unit vector)
        t_flat = t.ravel()
        result.translation_dir = tuple(float(x) for x in t_flat[:3])
    except Exception as e:
        logger.debug("Essential matrix decomposition failed: %s", e)

    result.motion_type = _classify_motion(result.ego_dx, result.ego_dy, mean_mag)
    return result


def _classify_motion(ego_dx: float, ego_dy: float, mean_mag: float) -> str:
    """Classify camera motion type from ego-motion vector."""
    ego_mag = math.sqrt(ego_dx ** 2 + ego_dy ** 2)

    if ego_mag < 1.5:
        return "static"

    # Ratio of horizontal to vertical motion
    if abs(ego_dx) > abs(ego_dy) * 2:
        return "panning"   # mostly horizontal → user turning head
    if abs(ego_dy) > abs(ego_dx) * 2:
        return "tilting"   # mostly vertical → looking up/down

    if mean_mag > 5.0:
        return "walking"   # significant overall motion
    return "moving"


def compensate_ego_motion(
    flow_vectors: list[tuple[float, float] | None],
    ego: EgoMotionResult,
) -> list[tuple[float, float] | None]:
    """Subtract ego-motion from per-object flow vectors.

    Returns the true object motion (relative to the world, not camera).
    """
    compensated = []
    for fv in flow_vectors:
        if fv is None:
            compensated.append(None)
        else:
            compensated.append((fv[0] - ego.ego_dx, fv[1] - ego.ego_dy))
    return compensated


def flow_to_velocity_mps(
    flow_dx: float,
    flow_dy: float,
    depth_relative: float | None,
    fps: float = 30.0,
    frame_width: int = 320,
    hfov_deg: float = 60.0,
) -> tuple[float, float, float] | None:
    """Convert pixel flow + monocular depth to approximate velocity in m/s.

    Uses pinhole camera model to project pixel motion into 3D space.
    Note: monocular depth is relative (0-1) → we use a pseudo-metric
    scale (0-1 maps to 0-10m).  For true metric, calibrate against
    known distances.

    Returns (vx_mps, vy_mps, speed_mps) or None if depth unavailable.
    """
    if depth_relative is None or depth_relative < 0.01:
        return None

    # Convert relative depth to pseudo-meters (0-1 → 0-10m)
    z_m = depth_relative * 10.0

    # Focal length from HFOV
    fx = frame_width / (2.0 * math.tan(math.radians(hfov_deg / 2)))

    # Pixel displacement per frame → world displacement
    # dx_world = (flow_dx * z_m) / fx
    dx_m = (flow_dx * z_m) / fx
    dy_m = (flow_dy * z_m) / fx  # assuming square pixels

    # Per frame → per second
    vx_mps = dx_m * fps
    vy_mps = dy_m * fps
    speed = math.sqrt(vx_mps ** 2 + vy_mps ** 2)

    return (round(vx_mps, 2), round(vy_mps, 2), round(speed, 2))
