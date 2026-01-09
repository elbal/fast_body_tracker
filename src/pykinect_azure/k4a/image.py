import numpy as np
from numpy import typing as npt
import cv2

from ._k4a_types import k4a_image_t
from . import _k4a
from . import k4a_const


class WrongImageFormat(Exception):
    pass


class Image:
    def __init__(self, image_handle: k4a_image_t):
        self._handle = image_handle

    def __del__(self):
        if self._handle:
            _k4a.K4aLib.k4a_image_release(self._handle)

    def handle(self):
        return self._handle

    @property
    def width(self) -> int:
        return int(_k4a.K4aLib.k4a_image_get_width_pixels(self._handle))

    @property
    def height(self) -> int:
        return int(_k4a.K4aLib.k4a_image_get_height_pixels(self._handle))

    @property
    def stride(self) -> int:
        return int(_k4a.K4aLib.k4a_image_get_stride_bytes(self._handle))

    @property
    def format(self) -> int:
        return int(_k4a.K4aLib.k4a_image_get_format(self._handle))

    @property
    def size(self) -> int:
        return int(_k4a.K4aLib.k4a_image_get_size(self._handle))

    @property
    def timestamp(self) -> int:
        return _k4a.K4aLib.k4a_image_get_device_timestamp_usec(self._handle)

    def to_numpy(self) -> npt.NDArray[np.uint8 | np.uint16 | np.int16]:
        buffer = np.ctypeslib.as_array(
            _k4a.K4aLib.k4a_image_get_buffer(self._handle), shape=(self.size,))

        # COLOR MJPG, decode with OpenCV.
        if self.format == k4a_const.K4A_IMAGE_FORMAT_COLOR_MJPG:
            return cv2.imdecode(buffer, -1)

        # NV12, YUV conversion.
        elif self.format == k4a_const.K4A_IMAGE_FORMAT_COLOR_NV12:
            yuv = buffer.reshape(int(self.height * 1.5), self.width)
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

        # YUY2, YUV conversion.
        elif self.format == k4a_const.K4A_IMAGE_FORMAT_COLOR_YUY2:
            yuv = buffer.reshape(self.height, self.width, 2)
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)

        # BGRA32.
        elif self.format == k4a_const.K4A_IMAGE_FORMAT_COLOR_BGRA32:
            stride = self.stride
            height = self.height
            width = self.width
            arr2d = buffer.reshape((height, stride))
            arr2d = arr2d[:, :width * 4]
            view3d = np.lib.stride_tricks.as_strided(
                arr2d, shape=(height, width, 4), strides=(stride, 4, 1))
            # Ensure contiguity for OpenCV plotting.
            return np.ascontiguousarray(view3d)

        # DEPTH16, IR16, CUSTOM16.
        elif self.format in (
                k4a_const.K4A_IMAGE_FORMAT_DEPTH16,
                k4a_const.K4A_IMAGE_FORMAT_IR16,
                k4a_const.K4A_IMAGE_FORMAT_CUSTOM16):
            arr16 = buffer.view("<u2")
            stride_elements = self.stride // 2
            arr16 = arr16.reshape((self.height, stride_elements))
            arr16 = arr16[:, :self.width]
            return np.ascontiguousarray(arr16)

        # CUSTOM8.
        elif self.format == k4a_const.K4A_IMAGE_FORMAT_CUSTOM8:
            stride_elements = self.stride
            arr8 = buffer.reshape((self.height, stride_elements))
            arr8 = arr8[:, :self.width]
            return np.ascontiguousarray(arr8)

        # CUSTOM.
        elif self.format == k4a_const.K4A_IMAGE_FORMAT_CUSTOM:
            arr = buffer.view("<i2").reshape((-1, 3))
            return arr

        else:
            raise WrongImageFormat(f"Unsupported format {self.format}.")
