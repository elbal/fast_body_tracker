import ctypes
import numpy as np
from numpy import typing as npt
import cv2
import matplotlib.pyplot as plt

from ..k4a import k4a_const
from ..k4a import Calibration
from ._k4abt_types import k4abt_body_t, k4a_float3
from . import kabt_const

JOINT_DTYPE = np.dtype([
    ("position", np.float32, 3), ("orientation", np.float32, 4),
    ("confidence", np.int32)
    ])

_cmap = plt.get_cmap("tab20")


class Body:
    def __init__(self, body_handle: k4abt_body_t):
        self._handle = body_handle
        joints = np.ctypeslib.as_array(
            self._handle.skeleton.joints,
            shape=(kabt_const.K4ABT_JOINT_COUNT,))
        self.joints_data = joints.view(JOINT_DTYPE)

        self.id = self._handle.id

    @property
    def positions(self) -> npt.NDArray[np.float32]:
        return self.joints_data["position"]

    @property
    def orientations(self) -> npt.NDArray[np.float32]:
        return self.joints_data["orientation"]

    @property
    def confidences(self) -> npt.NDArray[np.int32]:
        return self.joints_data['confidence']

    def get_2d_positions(
            self, calibration: Calibration,
            target_camera: int = k4a_const.K4A_CALIBRATION_TYPE_DEPTH) -> (
                npt.NDArray[np.float32]):
        positions_2d = [
            calibration.convert_3d_to_2d(
                position.ctypes.data_as(ctypes.POINTER(k4a_float3)).contents,
                k4a_const.K4A_CALIBRATION_TYPE_DEPTH, target_camera)
            for position in self.positions]
        return np.array(positions_2d, dtype=np.float32)


def draw_body(
        image: npt.NDArray[np.uint8], positions_2d: npt.NDArray[np.float32],
        body_id: int, only_segments: bool = False) -> npt.NDArray[np.uint8]:
    rgba = _cmap(body_id % 20)
    color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))

    positions = [tuple(position) for position in positions_2d.astype(np.int32)]

    for idx1, idx2 in kabt_const.K4ABT_SEGMENT_PAIRS:
        cv2.line(image, positions[idx1], positions[idx2], color, 2)

    if only_segments:
        return image

    for p in positions:
        cv2.circle(image, p, 3, color, 3)

    return image
