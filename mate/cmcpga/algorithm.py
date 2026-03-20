import numpy as np

from mate.cmcpga.clustering import build_camera_clusters
from mate.cmcpga.geometry import orient_to_action
from mate.cmcpga.models import CameraSnapshot, OrientationScore
from mate.cmcpga.scoring import compute_lambda, find_best_orientation


def cmcpga_decide(
    cameras: list[CameraSnapshot], target_positions: np.ndarray, epsilon_deg: float = 5.0
) -> np.ndarray:
    actions = np.zeros((len(cameras), 2), dtype=np.float64)
    if not cameras or len(target_positions) == 0:
        return actions

    clusters = build_camera_clusters(cameras)
    lambda_values = compute_lambda(clusters, cameras, target_positions)

    for members in clusters.values():
        covered_mask = np.zeros(len(target_positions), dtype=np.bool_)
        remaining = list(members)

        while remaining:
            best_choice: OrientationScore | None = None
            for camera_index in remaining:
                choice = find_best_orientation(
                    camera=cameras[camera_index],
                    target_positions=target_positions,
                    lambda_values=lambda_values,
                    covered_mask=covered_mask,
                    epsilon_deg=epsilon_deg,
                )
                if best_choice is None or choice.cp_value > best_choice.cp_value:
                    best_choice = choice
                    continue
                if best_choice is None or not np.isclose(choice.cp_value, best_choice.cp_value):
                    continue
                if choice.covered_mask.sum() > best_choice.covered_mask.sum():
                    best_choice = choice

            if best_choice is None:
                break

            camera = cameras[best_choice.camera_index]
            actions[camera.index] = orient_to_action(camera, best_choice.orientation)
            covered_mask |= best_choice.covered_mask
            remaining.remove(best_choice.camera_index)

            if not (~covered_mask).any():
                break

    return actions
