import numpy as np

from mate.cmcpga.models import CameraSnapshot


def normalize_angle_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def angle_diff_deg(target_angle: float, source_angle: float) -> float:
    return normalize_angle_deg(target_angle - source_angle)


def penalty_function(delta_deg: float) -> float:
    return 1.0 - abs(delta_deg) / 360.0


def bearing_deg(origin: np.ndarray, target: np.ndarray) -> float:
    delta = target - origin
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def covered_mask_for_orientation(
    camera: CameraSnapshot, orientation: float, target_positions: np.ndarray
) -> np.ndarray:
    if len(target_positions) == 0:
        return np.zeros(0, dtype=np.bool_)

    relative = target_positions - camera.position
    distances = np.linalg.norm(relative, axis=1)
    bearings = np.degrees(np.arctan2(relative[:, 1], relative[:, 0]))
    angle_diffs = np.abs([angle_diff_deg(beta, orientation) for beta in bearings])
    in_range = (distances >= camera.min_range) & (distances <= camera.range_limit)
    in_fov = angle_diffs <= camera.field_of_view / 2.0
    return in_range & in_fov


def orient_to_action(camera: CameraSnapshot, target_orientation: float) -> np.ndarray:
    delta = angle_diff_deg(target_orientation, camera.orientation)
    if abs(delta) < camera.rotation_step / 2.0:
        delta = 0.0
    else:
        delta = float(np.clip(delta, -camera.rotation_step, camera.rotation_step))
    return np.asarray([delta, 0.0], dtype=np.float64)
