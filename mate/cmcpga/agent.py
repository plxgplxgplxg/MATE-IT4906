import numpy as np

from mate.cmcpga.algorithm import cmcpga_decide
from mate.cmcpga.snapshots import snapshot_from_camera_entity


class CMCPGACameraAgent:
    def __init__(
        self, env, epsilon_deg: float = 5.0, active_targets_only: bool = True
    ) -> None:
        self.env = env
        self.epsilon_deg = float(epsilon_deg)
        self.active_targets_only = active_targets_only

    @property
    def num_cameras(self) -> int:
        return self.env.unwrapped.num_cameras

    def reset(self) -> None:
        return None

    def act(self, observations=None, info=None) -> np.ndarray:
        cameras = self._camera_snapshots()
        target_positions = self._target_positions()
        return cmcpga_decide(cameras, target_positions, epsilon_deg=self.epsilon_deg)

    def _camera_snapshots(self) -> list:
        return [
            snapshot_from_camera_entity(camera, index)
            for index, camera in enumerate(self.env.unwrapped.cameras)
        ]

    def _target_positions(self) -> np.ndarray:
        positions = []
        target_dones = getattr(self.env.unwrapped, "target_dones", None)
        for index, target in enumerate(self.env.unwrapped.targets):
            if self.active_targets_only and target_dones is not None and bool(target_dones[index]):
                continue
            positions.append(np.asarray(target.location, dtype=np.float64).copy())
        if not positions:
            return np.zeros((0, 2), dtype=np.float64)
        return np.vstack(positions)
