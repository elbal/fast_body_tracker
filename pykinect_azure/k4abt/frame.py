import ctypes
import numpy as np
from numpy import typing as npt
import cv2

from pykinect_azure.k4a._k4a_types import K4A_CALIBRATION_TYPE_DEPTH
from pykinect_azure.k4abt import _k4abt
from pykinect_azure.k4abt.body import Body
from pykinect_azure.k4abt.body2d import Body2d
from pykinect_azure.k4abt._k4abt_types import body_colors, k4abt_body_t
from pykinect_azure.k4a import Capture, Calibration, Image, Transformation


class Frame:
	def __init__(
			self, frame_handle: _k4abt.k4abt_frame_t, calibration: Calibration,
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
			target_camera: int = K4A_CALIBRATION_TYPE_DEPTH) -> Body2d:
		body_handle = k4abt_body_t()
		body_handle.id = _k4abt.k4abt_frame_get_body_id(self._handle, body_idx)
		body_handle.skeleton = self._get_body_skeleton(body_idx)

		return Body2d(body_handle, self.calibration, target_camera)

	def draw_bodies(
			self, destination_image: npt.NDArray[np.uint8],
			dest_camera: int = K4A_CALIBRATION_TYPE_DEPTH,
			only_segments: bool = False) -> npt.NDArray[np.uint8]:
		num_bodies = self.get_num_bodies()
		for body_id in range(num_bodies):
			destination_image = self.draw_body2d(
				destination_image, body_id, dest_camera, only_segments)

		return destination_image

	def draw_body2d(
			self, destination_image: npt.NDArray[np.uint8], body_idx: int = 0,
			dest_camera: int = K4A_CALIBRATION_TYPE_DEPTH,
			only_segments: bool = False) -> npt.NDArray[np.uint8]:
		return self.get_body2d(
			body_idx, dest_camera).draw(destination_image, only_segments)

	@property
	def timestamp(self) -> int:
		return _k4abt.k4abt_frame_get_device_timestamp_usec(self._handle)

	def get_segmentation_image(self) -> npt.NDArray[np.uint8]:
		body_index_map = self._get_body_index_map_object().to_numpy()
		return np.dstack(
			[cv2.LUT(body_index_map, body_colors[:, i]) for i in range(3)])

	def get_transformed_segmentation_image(
			self, capture: Capture) -> npt.NDArray[np.uint8]:
		depth_image = capture._get_depth_object()
		index_map = self.transformation.depth_image_to_color_camera_custom(
			depth_image, self._get_body_index_map_object())
		index_map = index_map.to_numpy()

		return np.dstack(
			[cv2.LUT(index_map, body_colors[:, i]) for i in range(3)])

	def _get_body_skeleton(self, index=0) -> _k4abt.k4abt_skeleton_t:
		skeleton = _k4abt.k4abt_skeleton_t()
		result_code = _k4abt.k4abt_frame_get_body_skeleton(
			self._handle, index, ctypes.byref(skeleton))
		if result_code != _k4abt.K4ABT_RESULT_SUCCEEDED:
			raise _k4abt.AzureKinectBodyTrackerException(
				"Body tracker get body skeleton failed.")

		return skeleton

	def _get_body_index_map_object(self) -> Image:
		return Image(_k4abt.k4abt_frame_get_body_index_map(self._handle))
