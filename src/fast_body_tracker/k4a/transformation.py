import ctypes
from dataclasses import dataclass

from ._k4a_types import k4a_image_t, k4a_transformation_t
from . import _k4a
from . import k4a_const
from .image import Image
from .calibration import Calibration

_STRIDE_BYTES_PER_PIXEL = {
    k4a_const.K4A_IMAGE_FORMAT_COLOR_BGRA32: 4,
    k4a_const.K4A_IMAGE_FORMAT_DEPTH16: 2,
    k4a_const.K4A_IMAGE_FORMAT_IR16: 2,
    k4a_const.K4A_IMAGE_FORMAT_CUSTOM: 6,
    k4a_const.K4A_IMAGE_FORMAT_CUSTOM8: 1,
    k4a_const.K4A_IMAGE_FORMAT_CUSTOM16: 2}


@dataclass
class Resolution:
    width: int
    height: int


class Transformation:
    def __init__(self, calibration: Calibration):
        self.calibration = calibration
        self._handle = _k4a.K4aLib.k4a_transformation_create(
            ctypes.byref(calibration.handle()))
        self.color_resolution = Resolution(
            calibration.handle().color_camera_calibration.resolution_width,
            calibration.handle().color_camera_calibration.resolution_height)
        self.depth_resolution = Resolution(
            calibration.handle().depth_camera_calibration.resolution_width,
            calibration.handle().depth_camera_calibration.resolution_height)

        depth_image_handle = self._create_image_handle(
            k4a_const.K4A_IMAGE_FORMAT_DEPTH16, self.color_resolution.width,
            self.color_resolution.height)
        self._depth_image_object = Image(depth_image_handle)
        self._invalid_val = ctypes.c_uint32(0)

    def __del__(self):
        if self._handle:
            _k4a.K4aLib.k4a_transformation_destroy(self._handle)

    def handle(self) -> k4a_transformation_t:
        return self._handle

    def depth_image_to_color_camera(
            self, depth_image_object: Image,
            transformed_image_object: Image | None = None) -> Image:
        if transformed_image_object is None:
            transformed_depth_handle = self._create_image_handle(
                depth_image_object.format, self.color_resolution.width,
                self.color_resolution.height)
            transformed_image_object = Image(transformed_depth_handle)

        _k4a.K4aLib.k4a_transformation_depth_image_to_color_camera(
            self._handle, depth_image_object.handle(),
            transformed_image_object.handle())

        return transformed_image_object

    def custom_image_to_color_camera(
            self, depth_image_object: Image, custom_image_object: Image,
            transformed_image_object: Image | None = None,
            interpolation: int = k4a_const.K4A_TRANSFORMATION_INTERPOLATION_TYPE_LINEAR) -> Image:
        if transformed_image_object is None:
            transformed_custom_handle = self._create_image_handle(
                custom_image_object.format, self.color_resolution.width,
                self.color_resolution.height)
            transformed_image_object = Image(transformed_custom_handle)

        _k4a.K4aLib.k4a_transformation_depth_image_to_color_camera_custom(
            self._handle, depth_image_object.handle(),
            custom_image_object.handle(), self._depth_image_object.handle(),
            transformed_image_object.handle(), interpolation,
            self._invalid_val)

        return transformed_image_object

    def color_image_to_depth_camera(
            self, depth_image_object: Image,
            color_image_object: Image,
            transformed_image_object: Image | None = None) -> Image:
        if transformed_image_object is None:
            transformed_image_handle = self._create_image_handle(
                k4a_const.K4A_IMAGE_FORMAT_COLOR_BGRA32,
                self.depth_resolution.width, self.depth_resolution.height)
            transformed_image_object = Image(transformed_image_handle)

        _k4a.K4aLib.k4a_transformation_color_image_to_depth_camera(
            self._handle, depth_image_object.handle(),
            color_image_object.handle(), transformed_image_object.handle())

        return transformed_image_object

    def depth_image_to_point_cloud(
            self, depth_image_object: Image,
            point_cloud_object: Image | None = None,
            calibration_type=k4a_const.K4A_CALIBRATION_TYPE_DEPTH, ) -> Image:
        if point_cloud_object is None:
            point_cloud_handle = self._create_image_handle(
                k4a_const.K4A_IMAGE_FORMAT_CUSTOM, depth_image_object.width,
                depth_image_object.height)
            point_cloud_object = Image(point_cloud_handle)

        _k4a.K4aLib.k4a_transformation_depth_image_to_point_cloud(
            self._handle, depth_image_object.handle(), calibration_type,
            point_cloud_object.handle())

        return point_cloud_object

    @staticmethod
    def _create_image_handle(
            image_format: int, width_pixels: int,
            height_pixels: int) -> k4a_image_t:
        stride_bytes = width_pixels * _STRIDE_BYTES_PER_PIXEL[image_format]

        image_handle = k4a_image_t()
        result_code = _k4a.K4aLib.k4a_image_create(
            image_format, width_pixels, height_pixels, stride_bytes,
            ctypes.byref(image_handle))
        if result_code != _k4a.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException("Create image failed.")

        return image_handle
