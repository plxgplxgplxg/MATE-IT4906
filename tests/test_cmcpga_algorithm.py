import unittest

import numpy as np

from mate.cmcpga.algorithm import cmcpga_decide
from mate.cmcpga.clustering import build_camera_clusters
from mate.cmcpga.models import CameraSnapshot


class CMCPGAAlgorithmTests(unittest.TestCase):
    def test_build_camera_clusters_groups_overlapping_cameras(self):
        cameras = [
            CameraSnapshot(0, np.array([0.0, 0.0]), 0.0, 60.0, 0.0, 5.0, 5.0),
            CameraSnapshot(1, np.array([6.0, 0.0]), 0.0, 60.0, 0.0, 5.0, 5.0),
            CameraSnapshot(2, np.array([20.0, 0.0]), 0.0, 60.0, 0.0, 5.0, 5.0),
        ]

        clusters = build_camera_clusters(cameras)
        members = sorted(sorted(value) for value in clusters.values())

        self.assertEqual(members, [[0, 1], [2]])

    def test_cmcpga_rotates_toward_target(self):
        cameras = [
            CameraSnapshot(0, np.array([0.0, 0.0]), 0.0, 60.0, 0.0, 10.0, 5.0),
        ]
        targets = np.array([[0.0, 5.0]], dtype=np.float64)

        action = cmcpga_decide(cameras, targets, epsilon_deg=90.0)

        self.assertEqual(action.shape, (1, 2))
        self.assertAlmostEqual(action[0, 0], 5.0)
        self.assertAlmostEqual(action[0, 1], 0.0)

    def test_cmcpga_returns_zero_when_no_targets(self):
        cameras = [
            CameraSnapshot(0, np.array([0.0, 0.0]), 0.0, 60.0, 0.0, 10.0, 5.0),
            CameraSnapshot(1, np.array([1.0, 0.0]), 0.0, 60.0, 0.0, 10.0, 5.0),
        ]

        action = cmcpga_decide(cameras, np.zeros((0, 2), dtype=np.float64), epsilon_deg=45.0)

        np.testing.assert_allclose(action, np.zeros((2, 2), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
