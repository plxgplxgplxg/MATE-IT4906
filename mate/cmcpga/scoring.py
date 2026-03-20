import numpy as np

from mate.cmcpga.geometry import angle_diff_deg, covered_mask_for_orientation, penalty_function
from mate.cmcpga.models import CameraSnapshot, OrientationScore


def compute_lambda(
    clusters: dict[int, list[int]], cameras: list[CameraSnapshot], target_positions: np.ndarray
) -> np.ndarray:
    if len(target_positions) == 0:
        return np.zeros(0, dtype=np.float64)

    lambda_values = np.zeros(len(target_positions), dtype=np.float64)
    for members in clusters.values():
        cluster_positions = np.vstack([cameras[index].position for index in members])
        cluster_ranges = np.asarray([cameras[index].range_limit for index in members], dtype=np.float64)
        deltas = target_positions[:, np.newaxis, :] - cluster_positions[np.newaxis, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        visible = (distances <= cluster_ranges[np.newaxis, :]).any(axis=1)
        lambda_values += visible.astype(np.float64)
    return np.maximum(lambda_values, 1.0)


def compute_cp_ratio(
    camera: CameraSnapshot,
    candidate_orientation: float,
    target_positions: np.ndarray,
    lambda_values: np.ndarray,
    covered_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    visible_mask = covered_mask_for_orientation(camera, candidate_orientation, target_positions)
    visible_mask &= ~covered_mask
    if not visible_mask.any():
        return 0.0, visible_mask

    delta = abs(angle_diff_deg(candidate_orientation, camera.orientation))
    penalty = max(penalty_function(delta), 1e-6)
    denominator = camera.field_of_view * camera.range_limit * penalty
    if denominator <= 1e-9:
        return 0.0, visible_mask

    gain = camera.rotation_step * np.sum(1.0 / lambda_values[visible_mask])
    return float(gain / denominator), visible_mask


def find_best_orientation(
    camera: CameraSnapshot,
    target_positions: np.ndarray,
    lambda_values: np.ndarray,
    covered_mask: np.ndarray,
    epsilon_deg: float = 5.0,
) -> OrientationScore:
    best = OrientationScore(
        camera_index=camera.index,
        cp_value=0.0,
        orientation=float(camera.orientation),
        covered_mask=np.zeros(len(target_positions), dtype=np.bool_),
    )

    if len(target_positions) == 0:
        return best

    candidates = np.arange(-180.0, 180.0, epsilon_deg, dtype=np.float64)
    if not np.isclose(candidates, camera.orientation).any():
        candidates = np.append(candidates, camera.orientation)

    for candidate in candidates:
        cp_value, visible_mask = compute_cp_ratio(
            camera, float(candidate), target_positions, lambda_values, covered_mask
        )
        if cp_value > best.cp_value:
            best = OrientationScore(
                camera_index=camera.index,
                cp_value=cp_value,
                orientation=float(candidate),
                covered_mask=visible_mask,
            )
            continue
        if not np.isclose(cp_value, best.cp_value):
            continue
        current_delta = abs(angle_diff_deg(candidate, camera.orientation))
        best_delta = abs(angle_diff_deg(best.orientation, camera.orientation))
        if current_delta < best_delta:
            best = OrientationScore(
                camera_index=camera.index,
                cp_value=cp_value,
                orientation=float(candidate),
                covered_mask=visible_mask,
            )

    return best
