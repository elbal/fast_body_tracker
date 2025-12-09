import cv2
import numpy as np
from numpy import typing as npt

from pykinect_azure.k4a import _k4a
from pykinect_azure.k4a.image import Image
from pykinect_azure.k4a.transformation import Transformation
from pykinect_azure.utils.postProcessing import smooth_depth_image


class Capture:
	def __init__(
			self, capture_handle: _k4a.k4a_capture_t,
			camera_transform: Transformation):
		self._handle = capture_handle
		self.camera_transform = camera_transform

	def __del__(self):
		if self._handle:
			_k4a.k4a_capture_release(self._handle)

	def handle(self) -> _k4a.k4a_capture_t:
		return self._handle

	def get_color_image(self) -> npt.NDArray[np.uint8]:
		return self._get_color_object().to_numpy()

	def get_transformed_color_image(self) -> npt.NDArray[np.uint8]:
		return self._get_transformed_color_object().to_numpy()

	def get_depth_image(self) -> npt.NDArray[np.uint16]:
		return self._get_depth_object().to_numpy()

	def get_colored_depth_image(self) -> npt.NDArray[np.uint8]:
		depth_image = self.get_depth_image()

		return self._color_depth_image(depth_image)

	def get_transformed_depth_image(self) -> npt.NDArray[np.uint16]:
		return self._get_transformed_depth_object().to_numpy()

	def get_transformed_colored_depth_image(self) -> npt.NDArray[np.uint8]:
		transformed_depth_image = self.get_transformed_depth_image()

		return self._color_depth_image(transformed_depth_image)

	def get_smooth_depth_image(
			self, maximum_hole_size: int = 10) -> npt.NDArray[np.uint16]:
		depth_image = self.get_depth_image()

		return smooth_depth_image(depth_image, maximum_hole_size)

	def get_smooth_colored_depth_image(
			self, maximum_hole_size=10) -> npt.NDArray[np.uint8]:
		smoothed_depth_image = self.get_smooth_depth_image(maximum_hole_size)

		return self._color_depth_image(smoothed_depth_image)

	def get_ir_image(self) -> npt.NDArray[np.uint16]:
		image_handle = _k4a.k4a_capture_get_ir_image(self._handle)

		return Image(image_handle).to_numpy()

	def get_pointcloud(
			self,
			calibration_type: int = _k4a.K4A_CALIBRATION_TYPE_DEPTH) -> npt.NDArray[np.int16]:
		points = self._get_pointcloud_object(calibration_type).to_numpy()
		points = points.reshape((-1, 3))

		return points

	def get_transformed_pointcloud(self) -> npt.NDArray[np.int16]:
		points = self._get_transformed_pointcloud_object().to_numpy()
		points = points.reshape((-1, 3))

		return points

	def _get_color_object(self) -> Image:
		image_handle = _k4a.k4a_capture_get_color_image(self._handle)

		return Image(image_handle)

	def _get_transformed_color_object(self) -> Image:
		depth_image = self._get_depth_object()
		color_image = self._get_color_object()

		return self.camera_transform.color_image_to_depth_camera(
			depth_image, color_image)

	def _get_depth_object(self) -> Image:
		image_handle = _k4a.k4a_capture_get_depth_image(self._handle)

		return Image(image_handle)

	def _get_transformed_depth_object(self) -> Image:
		depth_image = self._get_depth_object()

		return self.camera_transform.depth_image_to_color_camera(depth_image)

	def _get_pointcloud_object(
			self,
			calibration_type: int = _k4a.K4A_CALIBRATION_TYPE_DEPTH) -> Image:
		depth_image = self._get_depth_object()

		return self.camera_transform.depth_image_to_point_cloud(
			depth_image, calibration_type)

	def _get_transformed_pointcloud_object(self) -> Image:
		depth_image = self._get_transformed_depth_object()

		return self.camera_transform.depth_image_to_point_cloud(
			depth_image, _k4a.K4A_CALIBRATION_TYPE_COLOR)

	@staticmethod
	def _color_depth_image(
			depth_image: npt.NDArray[np.uint16 | np.int16]) -> npt.NDArray[np.uint8]:
		depth_color_image = cv2.convertScaleAbs(depth_image, alpha=0.05)
		depth_color_image = cv2.applyColorMap(
			depth_color_image, cv2.COLORMAP_CIVIDIS)

		return depth_color_image
