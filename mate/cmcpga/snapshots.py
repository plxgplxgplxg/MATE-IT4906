import numpy as np

from mate.cmcpga.models import CameraSnapshot


def snapshot_from_camera_entity(camera, index: int) -> CameraSnapshot:
    return CameraSnapshot(
        index=index,
        position=np.asarray(camera.location, dtype=np.float64).copy(),
        orientation=float(camera.orientation),
        field_of_view=float(camera.viewing_angle),
        min_range=0.0,
        range_limit=float(camera.sight_range),
        rotation_step=float(camera.rotation_step),
    )


def snapshot_from_camera_state(state) -> CameraSnapshot:
    return CameraSnapshot(
        index=int(state.index),
        position=np.asarray(state.location, dtype=np.float64).copy(),
        orientation=float(state.orientation),
        field_of_view=float(state.viewing_angle),
        min_range=0.0,
        range_limit=float(state.sight_range),
        rotation_step=float(state.rotation_step),
    )
