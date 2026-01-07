import ctypes
import numpy as np
from numpy import typing as npt

from ._k4a_types import (
    k4a_calibration_t, k4a_calibration_type_t, k4a_image_t, k4a_float2,
    k4a_float3)
from . import _k4a
from . import k4a_const


class Calibration:
    def __init__(self, calibration_handle: k4a_calibration_t):
        self._handle = calibration_handle

    def handle(self) -> k4a_calibration_t:
        return self._handle

    def get_k_matrix(self, camera: int) -> npt.NDArray[np.float32]:
        color_params = (
            self._handle.color_camera_calibration.intrinsics.parameters.param)
        depth_params = (
            self._handle.depth_camera_calibration.intrinsics.parameters.param)
        parameters_handler = {
            k4a_const.K4A_CALIBRATION_TYPE_COLOR: np.array([
                [color_params.fx, 0, color_params.cx],
                [0, color_params.fy, color_params.cy],
                [0, 0, 1]
            ]),
            k4a_const.K4A_CALIBRATION_TYPE_DEPTH: np.array([
                [depth_params.fx, 0, depth_params.cx],
                [0, depth_params.fy, depth_params.cy],
                [0, 0, 1]
            ])}

        return parameters_handler[camera]

    def get_dist_params(self, camera: int) -> npt.NDArray[np.float32]:
        if camera == k4a_const.K4A_CALIBRATION_TYPE_COLOR:
            intrinsics = self._handle.color_camera_calibration.intrinsics
        else:
            intrinsics = self._handle.depth_camera_calibration.intrinsics
        params = intrinsics.parameters.param

        if intrinsics.type == k4a_const.K4A_CALIBRATION_LENS_DISTORTION_MODEL_THETA:
            # WFOV depth (Fisheye) -> 4 parameters.
            return np.array(
                [params.k1, params.k2, params.k3, params.k4],
                dtype=np.float32)
        # NFOV depth (RATIONAL_6KT) -> 8 parameters (standard OpenCV).
        return np.array([
            params.k1, params.k2, params.p1, params.p2,
            params.k3, params.k4, params.k5, params.k6
        ], dtype=np.float32)

    def convert_3d_to_3d(
            self, source_point3d: k4a_float3,
            source_camera: k4a_calibration_type_t,
            target_camera: k4a_calibration_type_t) -> k4a_float3:
        conversion_fun = _k4a.K4aLib.k4a_calibration_3d_to_3d
        failure_message = "Failed to convert from 3D to 3D."
        target_point3d = k4a_float3()

        result_code = conversion_fun(
            ctypes.byref(self._handle), ctypes.byref(source_point3d),
            source_camera, target_camera, ctypes.byref(target_point3d))
        if result_code != k4a_const.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException(failure_message)

        return target_point3d

    def convert_2d_to_3d(
            self, source_point2d: k4a_float2, source_depth: float,
            source_camera: k4a_calibration_type_t,
            target_camera: k4a_calibration_type_t) -> k4a_float3:
        conversion_fun = _k4a.K4aLib.k4a_calibration_2d_to_3d
        failure_message = "Failed to convert from 2D to 3D."
        target_point3d = k4a_float3()
        valid = ctypes.c_int()

        result_code = conversion_fun(
            ctypes.byref(self._handle), ctypes.byref(source_point2d),
            source_depth, source_camera, target_camera,
            ctypes.byref(target_point3d), ctypes.byref(valid))
        if result_code != k4a_const.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException(failure_message)

        return target_point3d

    def convert_3d_to_2d(
            self, source_point3d: k4a_float3,
            source_camera: k4a_calibration_type_t,
            target_camera: k4a_calibration_type_t) -> k4a_float2:
        conversion_fun = _k4a.K4aLib.k4a_calibration_3d_to_2d
        failure_message = "Failed to convert from 3D to 2D."
        target_point2d = k4a_float2()
        valid = ctypes.c_int()

        result_code = conversion_fun(
            ctypes.byref(self._handle), ctypes.byref(source_point3d),
            source_camera, target_camera, ctypes.byref(target_point2d),
            ctypes.byref(valid))
        if result_code != k4a_const.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException(failure_message)

        return target_point2d

    def convert_2d_to_2d(
            self, source_point2d: k4a_float2, source_depth: float,
            source_camera: k4a_calibration_type_t,
            target_camera: k4a_calibration_type_t) -> k4a_float2:
        conversion_fun = _k4a.K4aLib.k4a_calibration_2d_to_2d
        failure_message = "Failed to convert from 2D to 2D."
        target_point2d = k4a_float2()
        valid = ctypes.c_int()

        result_code = conversion_fun(
            ctypes.byref(self._handle), ctypes.byref(source_point2d),
            source_depth, source_camera, target_camera,
            ctypes.byref(target_point2d), ctypes.byref(valid))
        if result_code != k4a_const.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException(failure_message)

        return target_point2d

    def convert_color_2d_to_depth_2d(
            self, source_point2d: k4a_float2,
            depth_image: k4a_image_t) -> k4a_float2:
        conversion_fun = _k4a.K4aLib.k4a_calibration_color_2d_to_depth_2d
        failure_message = "Failed to convert from Color 2D to Depth 2D."
        target_point2d = k4a_float2()
        valid = ctypes.c_int()

        result_code = conversion_fun(
            ctypes.byref(self._handle), ctypes.byref(source_point2d),
            depth_image, ctypes.byref(target_point2d), ctypes.byref(valid))
        if result_code != k4a_const.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException(failure_message)

        return target_point2d
