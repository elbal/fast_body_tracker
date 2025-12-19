import ctypes
import numpy as np
from numpy import typing as npt
import cv2
from dataclasses import dataclass

from pykinect_azure.k4a import _k4a
from pykinect_azure.k4a.image import Image
from pykinect_azure.k4a.calibration import Calibration

_STRIDE_BYTES_PER_PIXEL = {
	_k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32: 4,
	_k4a.K4A_IMAGE_FORMAT_DEPTH16: 2,
	_k4a.K4A_IMAGE_FORMAT_IR16: 2,
	_k4a.K4A_IMAGE_FORMAT_CUSTOM: 6,
	_k4a.K4A_IMAGE_FORMAT_CUSTOM8: 1,
	_k4a.K4A_IMAGE_FORMAT_CUSTOM16: 2}


@dataclass
class Resolution:
	width: int
	height: int


class Transformation:
	def __init__(self, calibration: Calibration):
		self.calibration = calibration
		self._handle = _k4a.k4a_transformation_create(
			ctypes.byref(calibration.handle()))
		self.color_resolution = Resolution(
			calibration.handle().color_camera_calibration.resolution_width,
			calibration.handle().color_camera_calibration.resolution_height)
		self.depth_resolution = Resolution(
			calibration.handle().depth_camera_calibration.resolution_width,
			calibration.handle().depth_camera_calibration.resolution_height)

	def __del__(self):
		if self._handle:
			_k4a.k4a_transformation_destroy(self._handle)

	def handle(self) -> _k4a.k4a_transformation_t:
		return self._handle

	def depth_image_to_color_camera(self, depth_image: Image) -> Image:
		transformed_depth_image_handle = self._create_image_handle(
			depth_image.format, self.color_resolution.width,
			self.color_resolution.height)

		_k4a.k4a_transformation_depth_image_to_color_camera(
			self._handle, depth_image.handle(), transformed_depth_image_handle)
		transformed_depth_image = Image(transformed_depth_image_handle)

		return transformed_depth_image

	def depth_and_custom_image_to_color_camera(
			self, depth_image: Image, custom_image: Image,
			interpolation=_k4a.K4A_TRANSFORMATION_INTERPOLATION_TYPE_LINEAR) -> Image:
		transformed_depth_image_handle = self._create_image_handle(
			_k4a.K4A_IMAGE_FORMAT_DEPTH16, self.color_resolution.width,
			self.color_resolution.height)
		transformed_custom_image_handle = self._create_image_handle(
			custom_image.format, self.color_resolution.width,
			self.color_resolution.height)
		invalid_custom_value = ctypes.c_uint32()

		_k4a.k4a_transformation_depth_image_to_color_camera_custom(
			self._handle, depth_image.handle(), custom_image.handle(),
			transformed_depth_image_handle, transformed_custom_image_handle,
			interpolation, invalid_custom_value)
		_ = Image(transformed_depth_image_handle)
		transformed_custom_image = Image(transformed_custom_image_handle)

		return transformed_custom_image

	def color_image_to_depth_camera(
			self, depth_image: Image, color_image: Image) -> Image:
		transformed_color_image_handle = self._create_image_handle(
			_k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32, self.depth_resolution.width,
			self.depth_resolution.height)

		_k4a.k4a_transformation_color_image_to_depth_camera(
			self._handle, depth_image.handle(), color_image.handle(),
			transformed_color_image_handle)
		transformed_color_image = Image(transformed_color_image_handle)

		return transformed_color_image

	def depth_image_to_point_cloud(
			self, depth_image: Image,
			calibration_type=_k4a.K4A_CALIBRATION_TYPE_DEPTH) -> Image:
		point_cloud_handle = self._create_image_handle(
			_k4a.K4A_IMAGE_FORMAT_CUSTOM, depth_image.width,
			depth_image.height)
		_k4a.k4a_transformation_depth_image_to_point_cloud(
			self._handle, depth_image.handle(), calibration_type,
			point_cloud_handle)
		point_cloud_image = Image(point_cloud_handle)

		return point_cloud_image

	@staticmethod
	def color_a_depth_image(
			depth_image: npt.NDArray[np.uint16 | np.int16]) -> npt.NDArray[
		np.uint8]:
		depth_8bit = cv2.convertScaleAbs(depth_image, alpha=0.05)
		cv2.bitwise_not(depth_8bit, dst=depth_8bit)
		depth_color_image = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)
		depth_color_image[depth_image == 0] = 0

		return depth_color_image

	@staticmethod
	def _create_image_handle(
			image_format: int, width_pixels: int,
			height_pixels: int) -> _k4a.k4a_image_t:
		stride_bytes = width_pixels * _STRIDE_BYTES_PER_PIXEL[image_format]

		image_handle = _k4a.k4a_image_t()
		result_code = _k4a.k4a_image_create(
			image_format, width_pixels, height_pixels, stride_bytes,
			ctypes.byref(image_handle))
		if result_code != _k4a.K4A_RESULT_SUCCEEDED:
			raise _k4a.AzureKinectSensorException("Create image failed.")

		return image_handle

	@staticmethod
	def smooth_a_depth_image(
			depth_image: Image, max_hole_size: int = 10) -> Image:
		"""
		Smoothes depth image by filling the holes using inpainting method.

		Parameters
		----------
		depth_image: Image:
			Original depth image.
		max_hole_size: int
			Maximum hole size to fill.

		Returns
		-------
		Image
			Smoothed depth image
		"""
		mask = np.zeros(depth_image.shape, dtype=np.uint8)
		mask[depth_image == 0] = 1

		kernel = np.ones((max_hole_size, max_hole_size), np.uint8)
		erosion = cv2.erode(mask, kernel, iterations=1)
		mask = mask - erosion

		smoothed_depth_image = cv2.inpaint(
			depth_image.astype(np.uint16), mask, max_hole_size, cv2.INPAINT_NS)

		return smoothed_depth_image
