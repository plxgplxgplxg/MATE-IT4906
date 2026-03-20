from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraSnapshot:
    index: int
    position: np.ndarray
    orientation: float
    field_of_view: float
    min_range: float
    range_limit: float
    rotation_step: float


@dataclass(frozen=True)
class OrientationScore:
    camera_index: int
    cp_value: float
    orientation: float
    covered_mask: np.ndarray
