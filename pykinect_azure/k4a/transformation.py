import ctypes
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

	def depth_image_to_color_camera_custom(
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
		xyz_image_handle = self._create_image_handle(
			_k4a.K4A_IMAGE_FORMAT_CUSTOM, depth_image.width,
			depth_image.height)

		_k4a.k4a_transformation_depth_image_to_point_cloud(
			self._handle, depth_image.handle(), calibration_type,
			xyz_image_handle)
		xyz_image = Image(xyz_image_handle)

		return xyz_image

	@staticmethod
	def _get_custom_bytes_per_pixel(custom_image: Image) -> int:
		if custom_image.format == _k4a.K4A_IMAGE_FORMAT_CUSTOM8:
			return 1
		else:
			return 2

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
