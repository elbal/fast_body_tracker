import ctypes
import numpy as np
from numpy import typing as npt
import cv2
import matplotlib.pyplot as plt

from ..k4a import k4a_const
from ..k4a import Capture, Calibration, Image, Transformation
from ._k4abt_types import k4abt_body_t, k4abt_frame_t, k4abt_skeleton_t
from . import _k4abt
from . import kabt_const
from .body import Body
from .body2d import Body2d

cmap = plt.get_cmap("tab20")
body_colors = np.zeros((256, 3), dtype=np.uint8)
for i in range(256):
	rgba = cmap(i % 20)
	body_colors[i] = [int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255)]


class Frame:
	def __init__(
			self, frame_handle: k4abt_frame_t, calibration: Calibration,
			transformation: Transformation):
		if frame_handle:
			self._handle = frame_handle
			self.calibration = calibration
			self.transformation = transformation

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
		body_handle.skeleton = self._get_body_skeleton(body_idx)

		return Body(body_handle)

	def get_body2d(
			self, body_idx: int = 0,
			target_camera: int = k4a_const.K4A_CALIBRATION_TYPE_DEPTH) -> Body2d:
		body_handle = k4abt_body_t()
		body_handle.id = _k4abt.k4abt_frame_get_body_id(self._handle, body_idx)
		body_handle.skeleton = self._get_body_skeleton(body_idx)

		return Body2d(body_handle, self.calibration, target_camera)

	def draw_bodies(
			self, destination_image: npt.NDArray[np.uint8],
			dest_camera: int = k4a_const.K4A_CALIBRATION_TYPE_DEPTH,
			only_segments: bool = False) -> npt.NDArray[np.uint8]:
		num_bodies = self.get_num_bodies()
		for body_id in range(num_bodies):
			destination_image = self.draw_body2d(
				destination_image, body_id, dest_camera, only_segments)

		return destination_image

	def draw_body2d(
			self, destination_image: npt.NDArray[np.uint8], body_idx: int = 0,
			dest_camera: int = k4a_const.K4A_CALIBRATION_TYPE_DEPTH,
			only_segments: bool = False) -> npt.NDArray[np.uint8]:
		return self.get_body2d(
			body_idx, dest_camera).draw(destination_image, only_segments)

	@property
	def timestamp(self) -> int:
		return _k4abt.k4abt_frame_get_device_timestamp_usec(self._handle)

	def get_segmentation_image(self) -> npt.NDArray[np.uint8]:
		body_index_map = self._get_body_index_map_object().to_numpy()
		return np.dstack(
			[cv2.LUT(body_index_map, body_colors[:, j]) for j in range(3)])

	def get_transformed_segmentation_image(
			self, capture: Capture) -> npt.NDArray[np.uint8]:
		depth_image = capture.get_depth_image_object()
		index_map = self.transformation.custom_image_to_color_camera(
			depth_image, self._get_body_index_map_object())
		index_map = index_map.to_numpy()

		return np.dstack(
			[cv2.LUT(index_map, body_colors[:, j]) for j in range(3)])

	def _get_body_skeleton(self, index=0) -> k4abt_skeleton_t:
		skeleton = k4abt_skeleton_t()
		result_code = _k4abt.k4abt_frame_get_body_skeleton(
			self._handle, index, ctypes.byref(skeleton))
		if result_code != kabt_const.K4ABT_RESULT_SUCCEEDED:
			raise _k4abt.AzureKinectBodyTrackerException(
				"Body tracker get body skeleton failed.")

		return skeleton

	def _get_body_index_map_object(self) -> Image:
		return Image(_k4abt.k4abt_frame_get_body_index_map(self._handle))
