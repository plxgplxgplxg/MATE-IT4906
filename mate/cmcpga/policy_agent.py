from typing import Optional

import numpy as np

from mate.agents.base import CameraAgentBase
from mate.cmcpga.algorithm import cmcpga_decide
from mate.cmcpga.snapshots import snapshot_from_camera_state


class CMCPGACameraPolicyAgent(CameraAgentBase):
    def __init__(self, seed: Optional[int] = None, epsilon_deg: float = 5.0) -> None:
        super().__init__(seed=seed)
        self.epsilon_deg = float(epsilon_deg)
        self.camera_states = {}
        self.target_states = []
        self.visible_target_indices = np.zeros(0, dtype=np.int64)

    def reset(self, observation: np.ndarray) -> None:
        super().reset(observation)
        self.camera_states = {self.index: self.state.copy()}
        self.target_states = [None] * self.num_targets
        self.visible_target_indices = np.zeros(0, dtype=np.int64)
        self._update_target_memory(observation)

    def observe(self, observation: np.ndarray, info: Optional[dict] = None) -> None:
        self.state, observation, info, _ = self.check_inputs(observation, info)
        self.camera_states[self.index] = self.state.copy()
        self._update_target_memory(observation)

    def act(
        self,
        observation: np.ndarray,
        info: Optional[dict] = None,
        deterministic: Optional[bool] = None,
    ) -> np.ndarray:
        self.state, observation, info, _ = self.check_inputs(observation, info)
        self.camera_states[self.index] = self.state.copy()
        self._update_target_memory(observation)
        camera_snapshots = self._camera_snapshots()
        target_positions = self._target_positions()
        joint_action = cmcpga_decide(
            camera_snapshots, target_positions, epsilon_deg=self.epsilon_deg
        )
        return joint_action[self.index]

    def send_responses(self):
        target_states = [
            self.target_states[index].copy()
            for index in self.visible_target_indices
            if self.target_states[index] is not None
        ]
        return (
            self.pack_message(
                recipient=None,
                content={"state": self.state.copy(), "target_states": target_states},
            ),
        )

    def receive_responses(self, messages):
        self.last_responses = tuple(messages)
        for message in self.last_responses:
            teammate_state = message.content.get("state")
            if teammate_state is not None:
                self.camera_states[teammate_state.index] = teammate_state.copy()
            for target_state in message.content.get("target_states", []):
                self.target_states[target_state.index] = target_state.copy()

    def _update_target_memory(self, observation: np.ndarray) -> None:
        target_states, tracked_bits = self.get_all_opponent_states(observation)
        self.visible_target_indices = np.flatnonzero(tracked_bits)
        for index in self.visible_target_indices:
            self.target_states[index] = target_states[index].copy()

    def _camera_snapshots(self):
        return [snapshot_from_camera_state(self.camera_states[index]) for index in range(self.num_cameras)]

    def _target_positions(self) -> np.ndarray:
        known_targets = [state.location.copy() for state in self.target_states if state is not None]
        if not known_targets:
            return np.zeros((0, 2), dtype=np.float64)
        return np.vstack(known_targets)
