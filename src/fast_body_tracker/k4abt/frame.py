import ctypes
import numpy as np
from numpy import typing as npt
import cv2
import matplotlib.pyplot as plt

from ..k4a import Image, Transformation
from ._k4abt_types import k4abt_body_t, k4abt_frame_t
from . import _k4abt
from . import kabt_const
from .body import Body

cmap = plt.get_cmap("tab20")
body_colors = np.zeros((256, 3), dtype=np.uint8)
for i in range(256):
    rgba = cmap(i % 20)
    body_colors[i] = [int(rgba[2] * 255), int(rgba[1] * 255),
                      int(rgba[0] * 255)]


class Frame:
    def __init__(self, frame_handle: k4abt_frame_t):
        self._handle = frame_handle

    def __del__(self):
        if self._handle:
            _k4abt.k4abt_frame_release(self._handle)

    def get_num_bodies(self) -> int:
        return _k4abt.k4abt_frame_get_num_bodies(self._handle)

    def get_bodies(self) -> list[Body]:
        num_bodies = self.get_num_bodies()
        bodies = []
        if num_bodies:
            for body_idx in range(num_bodies):
                bodies.append(self.get_body(body_idx))

        return bodies

    def get_body(self, body_idx: int = 0) -> Body:
        body_handle = k4abt_body_t()

        body_handle.id = _k4abt.k4abt_frame_get_body_id(self._handle, body_idx)

        result_code = _k4abt.k4abt_frame_get_body_skeleton(
            self._handle, body_idx, ctypes.byref(body_handle.skeleton))
        if result_code != kabt_const.K4ABT_RESULT_SUCCEEDED:
            raise _k4abt.AzureKinectBodyTrackerException(
                "Body tracker get body skeleton failed.")

        return Body(body_handle)

    def get_segmentation_image_object(self) -> Image:
        return Image(_k4abt.k4abt_frame_get_body_index_map(self._handle))

    @property
    def timestamp(self) -> int:
        return _k4abt.k4abt_frame_get_device_timestamp_usec(self._handle)


def colorize_segmentation_image(
        seg_image_object: Image) -> npt.NDArray[np.uint8]:
    seg_image = seg_image_object.to_numpy()

    return np.dstack(
        [cv2.LUT(seg_image, body_colors[:, j]) for j in range(3)])


def transform_segmentation_image(
        depth_image_object: Image, seg_image_object: Image,
        transformation: Transformation) -> npt.NDArray[np.uint8]:
    trans_seg_image = transformation.custom_image_to_color_camera(
        depth_image_object, seg_image_object)
    trans_seg_image = trans_seg_image.to_numpy()

    return np.dstack(
        [cv2.LUT(trans_seg_image, body_colors[:, j]) for j in range(3)])
