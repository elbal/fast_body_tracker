import ctypes

from . import _k4a_types
from .k4a_const import K4A_RESULT_SUCCEEDED


class AzureKinectSensorException(Exception):
    pass


class K4aLib:
    _dll = None

    k4a_device_get_installed_count = None
    k4a_device_open = None
    k4a_device_close = None
    k4a_device_get_capture = None
    k4a_device_get_imu_sample = None
    k4a_device_start_cameras = None
    k4a_device_stop_cameras = None
    k4a_device_start_imu = None
    k4a_device_stop_imu = None
    k4a_device_get_serialnum = None
    k4a_device_get_version = None
    k4a_device_get_color_control_capabilities = None
    k4a_device_get_color_control = None
    k4a_device_set_color_control = None
    k4a_device_get_raw_calibration = None
    k4a_device_get_calibration = None
    k4a_device_get_sync_jack = None

    k4a_capture_create = None
    k4a_capture_release = None
    k4a_capture_reference = None
    k4a_capture_get_color_image = None
    k4a_capture_get_depth_image = None
    k4a_capture_get_ir_image = None
    k4a_capture_set_color_image = None
    k4a_capture_set_depth_image = None
    k4a_capture_set_ir_image = None
    k4a_capture_set_temperature_c = None
    k4a_capture_get_temperature_c = None

    k4a_image_create = None
    k4a_image_create_from_buffer = None
    k4a_image_get_buffer = None
    k4a_image_get_size = None
    k4a_image_get_format = None
    k4a_image_get_width_pixels = None
    k4a_image_get_height_pixels = None
    k4a_image_get_stride_bytes = None
    k4a_image_get_device_timestamp_usec = None
    k4a_image_get_system_timestamp_nsec = None
    k4a_image_get_exposure_usec = None
    k4a_image_get_white_balance = None
    k4a_image_get_iso_speed = None
    k4a_image_set_device_timestamp_usec = None
    k4a_image_set_system_timestamp_nsec = None
    k4a_image_set_exposure_usec = None
    k4a_image_set_white_balance = None
    k4a_image_set_iso_speed = None
    k4a_image_reference = None
    k4a_image_release = None

    k4a_calibration_get_from_raw = None
    k4a_calibration_3d_to_3d = None
    k4a_calibration_2d_to_3d = None
    k4a_calibration_3d_to_2d = None
    k4a_calibration_2d_to_2d = None
    k4a_calibration_color_2d_to_depth_2d = None

    k4a_transformation_create = None
    k4a_transformation_destroy = None
    k4a_transformation_depth_image_to_color_camera = None
    k4a_transformation_depth_image_to_color_camera_custom = None
    k4a_transformation_color_image_to_depth_camera = None
    k4a_transformation_depth_image_to_point_cloud = None

    @classmethod
    def setup(cls, path):
        cls._dll = ctypes.CDLL(path)
        cls._bind_all()

    @classmethod
    def _bind(cls, name, restype, argtypes):
        func = getattr(cls._dll, name)
        func.restype = restype
        func.argtypes = argtypes
        setattr(cls, name, func)

    @classmethod
    def _bind_all(cls):
        cls._bind(
            "k4a_device_get_installed_count", ctypes.c_uint32, [])
        cls._bind(
            "k4a_device_open", ctypes.c_int,
            (ctypes.c_uint32, ctypes.POINTER(_k4a_types.k4a_device_t)))
        cls._bind(
            "k4a_device_close", None, (_k4a_types.k4a_device_t,))
        cls._bind(
            "k4a_device_get_capture", ctypes.c_int,
            (
                _k4a_types.k4a_device_t,
                ctypes.POINTER(_k4a_types.k4a_capture_t),
                ctypes.c_int32))
        cls._bind(
            "k4a_device_get_imu_sample", ctypes.c_int,
            (
                _k4a_types.k4a_device_t,
                ctypes.POINTER(_k4a_types.k4a_imu_sample_t),
                ctypes.c_int32))
        cls._bind(
            "k4a_device_start_cameras", _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_device_t,
                ctypes.POINTER(_k4a_types.k4a_device_configuration_t)))
        cls._bind(
            "k4a_device_stop_cameras", None, (_k4a_types.k4a_device_t,))
        cls._bind(
            "k4a_device_start_imu", _k4a_types.k4a_result_t,
            (_k4a_types.k4a_device_t,))
        cls._bind(
            "k4a_device_stop_imu", None, (_k4a_types.k4a_device_t,))
        cls._bind(
            "k4a_device_get_serialnum", _k4a_types.k4a_buffer_result_t,
            (_k4a_types.k4a_device_t, ctypes.c_char_p,
             ctypes.POINTER(ctypes.c_size_t)))
        cls._bind(
            "k4a_device_get_version", _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_device_t,
                ctypes.POINTER(_k4a_types.k4a_hardware_version_t)))
        cls._bind(
            "k4a_device_get_color_control_capabilities",
            _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_device_t,
                _k4a_types.k4a_color_control_command_t,
                ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(_k4a_types.k4a_color_control_mode_t)))
        cls._bind(
            "k4a_device_get_color_control", _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_device_t,
                _k4a_types.k4a_color_control_command_t,
                ctypes.POINTER(_k4a_types.k4a_color_control_mode_t),
                ctypes.POINTER(ctypes.c_int32)))
        cls._bind(
            "k4a_device_set_color_control", _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_device_t,
                _k4a_types.k4a_color_control_command_t,
                _k4a_types.k4a_color_control_mode_t, ctypes.c_int32))
        cls._bind(
            "k4a_device_get_raw_calibration", _k4a_types.k4a_buffer_result_t,
            (
                _k4a_types.k4a_device_t, ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_size_t)))
        cls._bind(
            "k4a_device_get_calibration", _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_device_t, _k4a_types.k4a_depth_mode_t,
                _k4a_types.k4a_color_resolution_t,
                ctypes.POINTER(_k4a_types.k4a_calibration_t)))
        cls._bind(
            "k4a_device_get_sync_jack", _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_device_t, ctypes.POINTER(ctypes.c_bool),
                ctypes.POINTER(ctypes.c_bool)))

        cls._bind(
            "k4a_capture_create", _k4a_types.k4a_result_t,
            (ctypes.POINTER(_k4a_types.k4a_capture_t),))
        cls._bind(
            "k4a_capture_release", None, (_k4a_types.k4a_capture_t,))
        cls._bind(
            "k4a_capture_reference", None, (_k4a_types.k4a_capture_t,))
        cls._bind(
            "k4a_capture_get_color_image", _k4a_types.k4a_image_t,
            (_k4a_types.k4a_capture_t,))
        cls._bind(
            "k4a_capture_get_depth_image", _k4a_types.k4a_image_t,
            (_k4a_types.k4a_capture_t,))
        cls._bind(
            "k4a_capture_get_ir_image", _k4a_types.k4a_image_t,
            (_k4a_types.k4a_capture_t,))
        cls._bind(
            "k4a_capture_set_color_image", None,
            (_k4a_types.k4a_capture_t, _k4a_types.k4a_image_t))
        cls._bind(
            "k4a_capture_set_depth_image", None,
            (_k4a_types.k4a_capture_t, _k4a_types.k4a_image_t))
        cls._bind(
            "k4a_capture_set_ir_image", None,
            (_k4a_types.k4a_capture_t, _k4a_types.k4a_image_t))
        cls._bind(
            "k4a_capture_set_temperature_c", None,
            (_k4a_types.k4a_capture_t, ctypes.c_float))
        cls._bind(
            "k4a_capture_get_temperature_c", ctypes.c_float,
            (_k4a_types.k4a_capture_t,))

        cls._bind(
            "k4a_image_create", _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_image_format_t, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.POINTER(_k4a_types.k4a_image_t)))
        cls._bind(
            "k4a_image_create_from_buffer", _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_image_format_t, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(_k4a_types.k4a_image_t)))
        cls._bind(
            "k4a_image_get_buffer", ctypes.POINTER(ctypes.c_uint8),
            (_k4a_types.k4a_image_t,))
        cls._bind(
            "k4a_image_get_size", ctypes.c_size_t, (_k4a_types.k4a_image_t,))
        cls._bind(
            "k4a_image_get_format", _k4a_types.k4a_image_format_t,
            (_k4a_types.k4a_image_t,))
        cls._bind(
            "k4a_image_get_width_pixels", ctypes.c_int,
            (_k4a_types.k4a_image_t,))
        cls._bind(
            "k4a_image_get_height_pixels", ctypes.c_int,
            (_k4a_types.k4a_image_t,))
        cls._bind(
            "k4a_image_get_stride_bytes", ctypes.c_int,
            (_k4a_types.k4a_image_t,))
        cls._bind(
            "k4a_image_get_device_timestamp_usec", ctypes.c_uint64,
            (_k4a_types.k4a_image_t,))
        cls._bind(
            "k4a_image_get_system_timestamp_nsec", ctypes.c_uint64,
            (_k4a_types.k4a_image_t,))
        cls._bind(
            "k4a_image_get_exposure_usec", ctypes.c_uint64,
            (_k4a_types.k4a_image_t,))
        cls._bind(
            "k4a_image_get_white_balance", ctypes.c_uint32,
            (_k4a_types.k4a_image_t,))
        cls._bind(
            "k4a_image_get_iso_speed", ctypes.c_uint32,
            (_k4a_types.k4a_image_t,))
        cls._bind(
            "k4a_image_set_device_timestamp_usec", None,
            (_k4a_types.k4a_image_t, ctypes.c_uint64))
        cls._bind(
            "k4a_image_set_system_timestamp_nsec", None,
            (_k4a_types.k4a_image_t, ctypes.c_uint64))
        cls._bind(
            "k4a_image_set_exposure_usec", None,
            (_k4a_types.k4a_image_t, ctypes.c_uint64))
        cls._bind(
            "k4a_image_set_white_balance", None,
            (_k4a_types.k4a_image_t, ctypes.c_uint32))
        cls._bind(
            "k4a_image_set_iso_speed", None,
            (_k4a_types.k4a_image_t, ctypes.c_uint32))
        cls._bind("k4a_image_reference", None, (_k4a_types.k4a_image_t,))
        cls._bind("k4a_image_release", None, (_k4a_types.k4a_image_t,))

        cls._bind(
            "k4a_calibration_get_from_raw", _k4a_types.k4a_result_t,
            (
                ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
                _k4a_types.k4a_depth_mode_t,
                _k4a_types.k4a_color_resolution_t,
                ctypes.POINTER(_k4a_types.k4a_calibration_t)))
        cls._bind(
            "k4a_calibration_3d_to_3d", _k4a_types.k4a_result_t,
            (
                ctypes.POINTER(_k4a_types.k4a_calibration_t),
                ctypes.POINTER(_k4a_types.k4a_float3),
                _k4a_types.k4a_calibration_type_t,
                _k4a_types.k4a_calibration_type_t,
                ctypes.POINTER(_k4a_types.k4a_float3)))
        cls._bind(
            "k4a_calibration_2d_to_3d", _k4a_types.k4a_result_t,
            (
                ctypes.POINTER(_k4a_types.k4a_calibration_t),
                ctypes.POINTER(_k4a_types.k4a_float2),
                ctypes.c_float, _k4a_types.k4a_calibration_type_t,
                _k4a_types.k4a_calibration_type_t,
                ctypes.POINTER(_k4a_types.k4a_float3),
                ctypes.POINTER(ctypes.c_int)))
        cls._bind(
            "k4a_calibration_3d_to_2d", _k4a_types.k4a_result_t,
            (
                ctypes.POINTER(_k4a_types.k4a_calibration_t),
                ctypes.POINTER(_k4a_types.k4a_float3),
                _k4a_types.k4a_calibration_type_t,
                _k4a_types.k4a_calibration_type_t,
                ctypes.POINTER(_k4a_types.k4a_float2),
                ctypes.POINTER(ctypes.c_int)))
        cls._bind(
            "k4a_calibration_2d_to_2d", _k4a_types.k4a_result_t,
            (
                ctypes.POINTER(_k4a_types.k4a_calibration_t),
                ctypes.POINTER(_k4a_types.k4a_float2),
                ctypes.c_float, _k4a_types.k4a_calibration_type_t,
                _k4a_types.k4a_calibration_type_t,
                ctypes.POINTER(_k4a_types.k4a_float2),
                ctypes.POINTER(ctypes.c_int)))
        cls._bind(
            "k4a_calibration_color_2d_to_depth_2d", _k4a_types.k4a_result_t,
            (
                ctypes.POINTER(_k4a_types.k4a_calibration_t),
                ctypes.POINTER(_k4a_types.k4a_float2), _k4a_types.k4a_image_t,
                ctypes.POINTER(_k4a_types.k4a_float2),
                ctypes.POINTER(ctypes.c_int)))

        cls._bind(
            "k4a_transformation_create", _k4a_types.k4a_transformation_t,
            (ctypes.POINTER(_k4a_types.k4a_calibration_t),))
        cls._bind(
            "k4a_transformation_destroy", None,
            (_k4a_types.k4a_transformation_t,))
        cls._bind(
            "k4a_transformation_depth_image_to_color_camera",
            _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_transformation_t, _k4a_types.k4a_image_t,
                _k4a_types.k4a_image_t))
        cls._bind(
            "k4a_transformation_depth_image_to_color_camera_custom",
            _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_transformation_t, _k4a_types.k4a_image_t,
                _k4a_types.k4a_image_t, _k4a_types.k4a_image_t,
                _k4a_types.k4a_image_t,
                _k4a_types.k4a_transformation_interpolation_type_t,
                ctypes.c_uint32))
        cls._bind(
            "k4a_transformation_color_image_to_depth_camera",
            _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_transformation_t, _k4a_types.k4a_image_t,
                _k4a_types.k4a_image_t, _k4a_types.k4a_image_t))
        cls._bind(
            "k4a_transformation_depth_image_to_point_cloud",
            _k4a_types.k4a_result_t,
            (
                _k4a_types.k4a_transformation_t, _k4a_types.k4a_image_t,
                _k4a_types.k4a_calibration_type_t, _k4a_types.k4a_image_t))


def verify(result, error):
    if result != K4A_RESULT_SUCCEEDED:
        raise AzureKinectSensorException(error)
