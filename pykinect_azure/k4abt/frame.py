import ctypes
import numpy as np
import cv2

from pykinect_azure.k4a._k4a_types import K4A_CALIBRATION_TYPE_DEPTH
from pykinect_azure.k4abt import _k4abt
from pykinect_azure.k4abt.body import Body
from pykinect_azure.k4abt.body2d import Body2d
from pykinect_azure.k4abt._k4abt_types import body_colors, k4abt_body_t
from pykinect_azure.k4a import Capture, Calibration, Image, Transformation


class Frame:
	def __init__(
			self, frame_handle, calibration: Calibration,
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

	def get_body_skeleton(self, index=0):
		skeleton = _k4abt.k4abt_skeleton_t()

		_k4abt.VERIFY(_k4abt.k4abt_frame_get_body_skeleton(self._handle, index, ctypes.byref(skeleton)), "Body tracker get body skeleton failed!")

		return skeleton

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
		body_handle.skeleton = self.get_body_skeleton(body_idx)

		return Body(body_handle)

	def get_body2d(
			self, body_idx: int = 0,
			target_camera: int = K4A_CALIBRATION_TYPE_DEPTH) -> Body2d:
		body_handle = k4abt_body_t()
		body_handle.id = _k4abt.k4abt_frame_get_body_id(self._handle, body_idx)
		body_handle.skeleton = self.get_body_skeleton(body_idx)

		return Body2d(body_handle, self.calibration, target_camera)

	def draw_bodies(
			self, destination_image: Image,
			dest_camera: int = K4A_CALIBRATION_TYPE_DEPTH,
			only_segments: bool = False):
		num_bodies = self.get_num_bodies()
		for body_id in range(num_bodies):
			destination_image = self.draw_body2d(
				destination_image, body_id, dest_camera, only_segments)

		return destination_image

	def draw_body2d(
			self, destination_image: Image, body_idx: int = 0,
			dest_camera: int = K4A_CALIBRATION_TYPE_DEPTH,
			only_segments: bool = False):
		return self.get_body2d(
			body_idx, dest_camera).draw(destination_image, only_segments)

	def get_device_timestamp_usec(self):
		return _k4abt.k4abt_frame_get_device_timestamp_usec(self._handle)

	def get_body_index_map(self):
		return Image(_k4abt.k4abt_frame_get_body_index_map(self._handle))

	def get_body_index_map_image(self):
		return self.get_body_index_map().to_numpy()

	def get_transformed_body_index_map(self):
		depth_image = self.get_capture()._get_depth_object()
		return self.transformation.depth_image_to_color_camera_custom(depth_image, self.get_body_index_map())

	def get_transformed_body_index_map_image(self):
		transformed_body_index_map =self.get_transformed_body_index_map()
		return transformed_body_index_map.to_numpy()

	def get_segmentation_image(self):
		body_index_map = self.get_body_index_map_image()
		return np.dstack([cv2.LUT(body_index_map, body_colors[:,i]) for i in range(3)])

	def get_transformed_segmentation_image(self):
		ret, transformed_body_index_map = self.get_transformed_body_index_map_image()
		return ret, np.dstack([cv2.LUT(transformed_body_index_map, body_colors[:,i]) for i in range(3)])
		
	def get_capture(self):
		return Capture(_k4abt.k4abt_frame_get_capture(self._handle), self.calibration._handle)
