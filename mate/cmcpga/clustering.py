from collections import defaultdict

import numpy as np

from mate.cmcpga.models import CameraSnapshot


class DisjointSetUnion:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))
        self.ranks = [0] * size

    def find(self, value: int) -> int:
        parent = self.parents[value]
        if parent != value:
            self.parents[value] = self.find(parent)
        return self.parents[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.ranks[left_root] < self.ranks[right_root]:
            left_root, right_root = right_root, left_root
        self.parents[right_root] = left_root
        if self.ranks[left_root] == self.ranks[right_root]:
            self.ranks[left_root] += 1

    def clusters(self) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.parents)):
            grouped[self.find(index)].append(index)
        return {root: members for root, members in grouped.items()}


def build_camera_clusters(cameras: list[CameraSnapshot]) -> dict[int, list[int]]:
    dsu = DisjointSetUnion(len(cameras))
    for left in range(len(cameras)):
        for right in range(left + 1, len(cameras)):
            distance = np.linalg.norm(cameras[left].position - cameras[right].position)
            if distance < cameras[left].range_limit + cameras[right].range_limit:
                dsu.union(left, right)
    return dsu.clusters()
