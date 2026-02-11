"""Trajectory prediction & collision detection for tracked objects.

Uses constant-velocity / constant-acceleration Kalman filter to forecast
object positions 1–3 seconds ahead.  Combined with depth estimation,
this enables proactive alerts:

  "Sir, bicycle approaching from left at 8 km/h — potential collision in 2.4 seconds"

Inspired by Tesla FSD's occupancy flow prediction and SpaceX Dragon's
trajectory forecasting for docking approach.

Memory: ~0 extra — pure NumPy computation per tracked object.
"""

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PredictedTrajectory:
    """Forecasted trajectory for a tracked object."""

    track_id: int = 0
    class_name: str = ""

    # Current state
    position_px: tuple[float, float] = (0.0, 0.0)   # center (cx, cy) in pixels
    velocity_px: tuple[float, float] = (0.0, 0.0)    # pixels/sec
    velocity_mps: tuple[float, float, float] | None = None  # (vx, vy, speed) m/s
    depth_m: float | None = None                      # estimated distance in meters

    # Predicted future positions (list of (cx, cy, t_sec))
    waypoints: list[tuple[float, float, float]] = field(default_factory=list)

    # Collision risk
    collision_risk: float = 0.0        # 0.0-1.0 probability of entering danger zone
    time_to_collision: float | None = None  # seconds until potential collision, or None
    collision_direction: str = ""       # "left", "right", "ahead", "behind"

    # Behaviour classification
    behaviour: str = "stationary"       # stationary, approaching, receding, crossing, orbiting


@dataclass
class CollisionAlert:
    """Proactive collision/proximity alert for the orchestrator."""

    track_id: int = 0
    class_name: str = ""
    speed_mps: float = 0.0
    distance_m: float = 0.0
    time_to_collision: float = 0.0
    direction: str = ""
    severity: str = "notice"  # notice, warning, critical
    message: str = ""


# ── Trajectory predictor ──────────────────────────────────────────────


class TrajectoryPredictor:
    """Predict future positions and detect collision risks.

    Call ``predict_all()`` each frame with tracked objects, flow data,
    and depth info to get trajectories and collision alerts.

    Parameters
    ----------
    prediction_horizon_sec : float
        How far ahead to predict (seconds).
    prediction_steps : int
        Number of waypoints in the prediction horizon.
    collision_zone_m : float
        Radius of "danger zone" around camera (meters).
    approach_angle_deg : float
        Max angle from camera center-line to count as "approaching".
    """

    def __init__(
        self,
        prediction_horizon_sec: float = 3.0,
        prediction_steps: int = 6,
        collision_zone_m: float = 2.0,
        approach_angle_deg: float = 45.0,
    ):
        self.horizon = prediction_horizon_sec
        self.steps = prediction_steps
        self.collision_zone_m = collision_zone_m
        self.approach_angle = approach_angle_deg

        # Per-track acceleration estimator (track_id → prev velocity)
        self._prev_velocities: dict[int, tuple[float, float]] = {}

    def predict_all(
        self,
        tracked_objects: list,
        flow_vectors: list[tuple[float, float] | None] | None = None,
        depth_values: list[float | None] | None = None,
        velocity_mps_list: list[tuple[float, float, float] | None] | None = None,
        frame_size: tuple[int, int] = (320, 240),
        fps: float = 30.0,
    ) -> tuple[list[PredictedTrajectory], list[CollisionAlert]]:
        """Predict trajectories and detect collisions for all tracked objects.

        Parameters
        ----------
        tracked_objects : list of TrackedObject (from ByteTrackLite)
        flow_vectors : per-object (dx, dy) ego-compensated flow, or None
        depth_values : per-object relative depth (0-1), or None
        velocity_mps_list : per-object (vx, vy, speed) in m/s, or None
        frame_size : (W, H) for reference
        fps : camera FPS for velocity conversion

        Returns
        -------
        (trajectories, alerts) — list of PredictedTrajectory and CollisionAlert
        """
        trajectories = []
        alerts = []
        fw, fh = frame_size

        for i, t in enumerate(tracked_objects):
            track_id = getattr(t, "track_id", i)
            class_name = getattr(t, "class_name", "object")
            xyxy = getattr(t, "xyxy", [0, 0, 0, 0])
            vel = getattr(t, "velocity", [0.0, 0.0])

            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0

            # Prefer flow-based velocity if available
            if flow_vectors and i < len(flow_vectors) and flow_vectors[i] is not None:
                vx_px = flow_vectors[i][0] * fps  # per-frame → per-sec
                vy_px = flow_vectors[i][1] * fps
            else:
                vx_px = vel[0] if len(vel) > 0 else 0.0
                vy_px = vel[1] if len(vel) > 1 else 0.0

            # Estimate acceleration (simple finite difference)
            prev_vel = self._prev_velocities.get(track_id, (vx_px, vy_px))
            ax = (vx_px - prev_vel[0]) * 0.3  # dampened acceleration
            ay = (vy_px - prev_vel[1]) * 0.3
            self._prev_velocities[track_id] = (vx_px, vy_px)

            # Depth
            depth_rel = None
            if depth_values and i < len(depth_values):
                depth_rel = depth_values[i]
            elif hasattr(t, "depth") and t.depth is not None:
                depth_rel = t.depth

            depth_m = depth_rel * 10.0 if depth_rel is not None else None

            # Velocity in m/s
            vel_mps = None
            if velocity_mps_list and i < len(velocity_mps_list):
                vel_mps = velocity_mps_list[i]

            # ── Predict waypoints ─────────────────────────────────
            dt = self.horizon / self.steps
            waypoints = []

            for step in range(1, self.steps + 1):
                t_sec = step * dt
                # Constant acceleration model
                px = cx + vx_px * t_sec + 0.5 * ax * t_sec ** 2
                py = cy + vy_px * t_sec + 0.5 * ay * t_sec ** 2
                waypoints.append((round(px, 1), round(py, 1), round(t_sec, 2)))

            # ── Classify behaviour ────────────────────────────────
            speed_px = math.sqrt(vx_px ** 2 + vy_px ** 2)
            behaviour = _classify_behaviour(
                vx_px, vy_px, cx, cy, fw, fh, speed_px
            )

            # ── Collision risk ────────────────────────────────────
            collision_risk = 0.0
            ttc: float | None = None
            direction = ""

            if depth_m is not None and vel_mps is not None:
                speed_mps = vel_mps[2]
                # Simple time-to-collision: distance / closing speed
                # "Closing speed" = component of velocity toward camera
                # In monocular, objects approaching have decreasing depth
                # We approximate: if object is getting bigger (expanding bbox)
                # or has significant speed toward camera center → approaching

                if behaviour == "approaching" and speed_mps > 0.1:
                    ttc = depth_m / speed_mps if speed_mps > 0 else None
                    if ttc is not None and ttc < self.horizon:
                        collision_risk = min(1.0, self.collision_zone_m / max(depth_m, 0.1))

                # Determine approach direction
                if cx < fw * 0.33:
                    direction = "left"
                elif cx > fw * 0.67:
                    direction = "right"
                else:
                    direction = "ahead"

            traj = PredictedTrajectory(
                track_id=track_id,
                class_name=class_name,
                position_px=(cx, cy),
                velocity_px=(vx_px, vy_px),
                velocity_mps=vel_mps,
                depth_m=depth_m,
                waypoints=waypoints,
                collision_risk=collision_risk,
                time_to_collision=ttc,
                collision_direction=direction,
                behaviour=behaviour,
            )
            trajectories.append(traj)

            # ── Generate alert if needed ──────────────────────────
            if ttc is not None and ttc < self.horizon and collision_risk > 0.2:
                speed_display = vel_mps[2] if vel_mps else 0
                alert = _build_alert(
                    track_id, class_name, speed_display,
                    depth_m or 0, ttc, direction,
                )
                if alert is not None:
                    alerts.append(alert)

        # Clean up stale velocity history
        active_ids = {
            getattr(t, "track_id", i) for i, t in enumerate(tracked_objects)
        }
        self._prev_velocities = {
            k: v for k, v in self._prev_velocities.items() if k in active_ids
        }

        return trajectories, alerts

    def reset(self) -> None:
        """Clear prediction state."""
        self._prev_velocities.clear()


# ── Helper functions ──────────────────────────────────────────────────


def _classify_behaviour(
    vx: float, vy: float, cx: float, cy: float,
    fw: int, fh: int, speed: float,
) -> str:
    """Classify object motion behaviour in the image plane."""
    if speed < 10:  # pixels/sec threshold for "stationary"
        return "stationary"

    # Camera center
    cam_cx, cam_cy = fw / 2.0, fh / 2.0

    # Vector from object to camera center
    to_cam_x = cam_cx - cx
    to_cam_y = cam_cy - cy
    to_cam_mag = math.sqrt(to_cam_x ** 2 + to_cam_y ** 2)

    if to_cam_mag < 1:
        return "orbiting"

    # Dot product of velocity with direction-to-camera
    dot = (vx * to_cam_x + vy * to_cam_y) / to_cam_mag
    # Cross product magnitude (lateral component)
    cross = abs(vx * to_cam_y - vy * to_cam_x) / to_cam_mag

    if dot > speed * 0.5:
        return "approaching"
    elif dot < -speed * 0.5:
        return "receding"
    elif cross > speed * 0.5:
        return "crossing"
    else:
        return "moving"


def _build_alert(
    track_id: int,
    class_name: str,
    speed_mps: float,
    distance_m: float,
    ttc: float,
    direction: str,
) -> CollisionAlert | None:
    """Build a collision alert with severity and natural language message."""
    if ttc <= 0:
        return None

    # Convert m/s to km/h for display
    speed_kmh = speed_mps * 3.6

    # Determine severity
    if ttc < 1.0 and distance_m < 2.0:
        severity = "critical"
    elif ttc < 2.0 and distance_m < 4.0:
        severity = "warning"
    elif ttc < 3.0:
        severity = "notice"
    else:
        return None  # not urgent enough

    # Build natural language message (Jarvis-style)
    dir_phrase = f"from the {direction}" if direction else ""
    msg = (
        f"Sir, {class_name} {dir_phrase} at {speed_kmh:.0f} km/h — "
        f"approximately {distance_m:.1f} meters away, "
        f"potential collision in {ttc:.1f} seconds."
    )

    return CollisionAlert(
        track_id=track_id,
        class_name=class_name,
        speed_mps=speed_mps,
        distance_m=distance_m,
        time_to_collision=ttc,
        direction=direction,
        severity=severity,
        message=msg,
    )


def format_trajectory_summary(
    trajectories: list[PredictedTrajectory],
    alerts: list[CollisionAlert],
    ego_motion_type: str = "static",
) -> str:
    """Format trajectory data as concise text for LLM context injection.

    Example: "Ego: walking forward. 2 objects tracked: person approaching
    at 1.2 m/s (3.8m, collision in 2.4s), car crossing left at 5.1 m/s."
    """
    parts = []

    # Ego-motion summary
    if ego_motion_type != "static":
        parts.append(f"Ego: {ego_motion_type}")

    # Object summaries (only moving or close objects)
    moving = [t for t in trajectories if t.behaviour != "stationary"]
    if moving:
        obj_parts = []
        for t in moving[:5]:  # cap at 5 most interesting
            desc = f"{t.class_name} {t.behaviour}"
            if t.velocity_mps:
                desc += f" at {t.velocity_mps[2]:.1f}m/s"
            if t.depth_m is not None:
                desc += f" ({t.depth_m:.1f}m)"
            if t.time_to_collision is not None:
                desc += f" [collision {t.time_to_collision:.1f}s]"
            obj_parts.append(desc)
        parts.append(f"{len(moving)} moving: " + ", ".join(obj_parts))

    # Alerts
    if alerts:
        for a in alerts[:2]:  # cap at 2 most urgent
            parts.append(f"[{a.severity.upper()}] {a.message}")

    return ". ".join(parts) if parts else ""
