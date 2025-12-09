import numpy as np
from numpy import typing as npt
import cv2

from pykinect_azure.k4a import _k4a

_IMAGE_FORMATS_HANDLER = {
	_k4a.K4A_IMAGE_FORMAT_COLOR_MJPG: (
		np.uint8, None, None, lambda img: cv2.imdecode(img, -1)),
	_k4a.K4A_IMAGE_FORMAT_COLOR_NV12: (
		np.uint8, 1.5, 1,
		lambda img: cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)),
	_k4a.K4A_IMAGE_FORMAT_COLOR_YUY2: (
		np.uint8, 1, 2, lambda img: cv2.cvtColor(
			img.reshape(img.shape[0], -1, 2), cv2.COLOR_YUV2BGR_YUY2)),
	_k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32: (
		np.uint8, 1, 4, lambda img: img.reshape(img.shape[0], -1, 4).copy()),
	_k4a.K4A_IMAGE_FORMAT_DEPTH16: ("<u2", 1, 1, None),
	_k4a.K4A_IMAGE_FORMAT_IR16: ("<u2", 1, 1, None),
	_k4a.K4A_IMAGE_FORMAT_CUSTOM8: ("<u1", 1, 1, None),
	_k4a.K4A_IMAGE_FORMAT_CUSTOM16: ("<u2", 1, 1, None),
	_k4a.K4A_IMAGE_FORMAT_CUSTOM: ("<i2", 1, 1, None)}


class WrongImageFormat(Exception):
	pass


class Image:
	def __init__(self, image_handle: _k4a.k4a_image_t):
		self._handle = image_handle
		self.buffer_pointer = _k4a.k4a_image_get_buffer(self._handle)

	def __del__(self):
		if self._handle:
			_k4a.k4a_image_release(self._handle)

	def handle(self):
		return self._handle

	@property
	def width(self) -> int:
		return int(_k4a.k4a_image_get_width_pixels(self._handle))

	@property
	def height(self) -> int:
		return int(_k4a.k4a_image_get_height_pixels(self._handle))

	@property
	def stride(self) -> int:
		return int(_k4a.k4a_image_get_stride_bytes(self._handle))

	@property
	def format(self) -> int:
		return int(_k4a.k4a_image_get_format(self._handle))

	@property
	def size(self) -> int:
		return int(_k4a.k4a_image_get_size(self._handle))

	@property
	def device_timestamp_usec(self) -> int:
		return _k4a.k4a_image_get_device_timestamp_usec(self._handle)

	def to_numpy(self) -> tuple[
			bool, npt.NDArray[np.uint8 | np.uint16 | np.int16]]:
		if self.format not in _IMAGE_FORMATS_HANDLER:
			raise WrongImageFormat("The image format is not supported.")
		dtype, h_scale, w_scale, process_fn = _IMAGE_FORMATS_HANDLER[self.format]
		buffer_array = np.ctypeslib.as_array(
			_k4a.k4a_image_get_buffer(self._handle), shape=(self.size,))
		image = np.frombuffer(buffer_array, dtype=dtype)
		if h_scale is None:
			return True, process_fn(image)

		rows = int(self.height * h_scale)
		itemsize = np.dtype(dtype).itemsize
		stride_elements = self.stride // itemsize
		image = image.reshape(rows, stride_elements)
		image = image[:, :int(self.width * w_scale)]
		if process_fn:
			image = process_fn(image)
		else:
			image = image.reshape(self.height, self.width)
			image = image.copy()

		return True, image
