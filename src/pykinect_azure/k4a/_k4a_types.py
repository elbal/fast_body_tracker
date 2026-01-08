import ctypes

from . import k4a_const

k4a_buffer_result_t = ctypes.c_int
k4a_calibration_model_type_t = ctypes.c_int
k4a_calibration_type_t = ctypes.c_int
k4a_color_control_command_t = ctypes.c_int
k4a_color_control_mode_t = ctypes.c_int
k4a_color_resolution_t = ctypes.c_int
k4a_depth_mode_t = ctypes.c_int
k4a_firmware_build_t = ctypes.c_int
k4a_firmware_signature_t = ctypes.c_int
k4a_fps_t = ctypes.c_int
k4a_image_format_t = ctypes.c_int
k4a_log_level_t = ctypes.c_int
k4a_result_t = ctypes.c_int
k4a_transformation_interpolation_type_t = ctypes.c_int
k4a_wait_result_t = ctypes.c_int
k4a_wired_sync_mode_t = ctypes.c_int


class k4a_float3(ctypes.Array):
    _type_ = ctypes.c_float
    _length_ = 3


class k4a_float2(ctypes.Array):
    _type_ = ctypes.c_float
    _length_ = 2


class _k4a_device_t(ctypes.Structure):
    _fields_ = [("_rsvd", ctypes.c_size_t)]


k4a_device_t = ctypes.POINTER(_k4a_device_t)


class _k4a_capture_t(ctypes.Structure):
    _fields_ = [("_rsvd", ctypes.c_size_t)]


k4a_capture_t = ctypes.POINTER(_k4a_capture_t)


class _k4a_image_t(ctypes.Structure):
    _fields_ = [("_rsvd", ctypes.c_size_t)]


k4a_image_t = ctypes.POINTER(_k4a_image_t)


class _k4a_transformation_t(ctypes.Structure):
    _fields_ = [("_rsvd", ctypes.c_size_t)]


k4a_transformation_t = ctypes.POINTER(_k4a_transformation_t)


class k4a_device_configuration_t(ctypes.Structure):
    _fields_ = [
        ("color_format", ctypes.c_int),
        ("color_resolution", ctypes.c_int),
        ("depth_mode", ctypes.c_int),
        ("camera_fps", ctypes.c_int),
        ("synchronized_images_only", ctypes.c_bool),
        ("depth_delay_off_color_usec", ctypes.c_int32),
        ("wired_sync_mode", ctypes.c_int),
        ("subordinate_delay_off_master_usec", ctypes.c_uint32),
        ("disable_streaming_indicator", ctypes.c_bool)
    ]


class k4a_calibration_extrinsics_t(ctypes.Structure):
    _fields_ = [
        ("rotation", ctypes.c_float * 9),
        ("translation", ctypes.c_float * 3), ]


class k4a_param(ctypes.Structure):
    _fields_ = [
        ("cx", ctypes.c_float), ("cy", ctypes.c_float),
        ("fx", ctypes.c_float), ("fy", ctypes.c_float),
        ("k1", ctypes.c_float), ("k2", ctypes.c_float),
        ("k3", ctypes.c_float), ("k4", ctypes.c_float),
        ("k5", ctypes.c_float), ("k6", ctypes.c_float),
        ("codx", ctypes.c_float), ("cody", ctypes.c_float),
        ("p2", ctypes.c_float), ("p1", ctypes.c_float),
        ("metric_radius", ctypes.c_float)
    ]


class k4a_calibration_intrinsic_parameters_t(ctypes.Union):
    _fields_ = [("param", k4a_param), ("v", ctypes.c_float * 15)]


class k4a_calibration_intrinsics_t(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("parameter_count", ctypes.c_uint),
        ("parameters", k4a_calibration_intrinsic_parameters_t)
    ]


class k4a_calibration_camera_t(ctypes.Structure):
    _fields_ = [
        ("extrinsics", k4a_calibration_extrinsics_t),
        ("intrinsics", k4a_calibration_intrinsics_t),
        ("resolution_width", ctypes.c_int),
        ("resolution_height", ctypes.c_int),
        ("metric_radius", ctypes.c_float)
    ]


class k4a_calibration_t(ctypes.Structure):
    _fields_ = [
        ("depth_camera_calibration", k4a_calibration_camera_t),
        ("color_camera_calibration", k4a_calibration_camera_t),
        ("extrinsics", (
                k4a_calibration_extrinsics_t
                * k4a_const.K4A_CALIBRATION_TYPE_NUM
                * k4a_const.K4A_CALIBRATION_TYPE_NUM)),
        ("depth_mode", ctypes.c_int),
        ("color_resolution", ctypes.c_int)
    ]


class k4a_version_t(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("iteration", ctypes.c_uint32)
    ]


class k4a_hardware_version_t(ctypes.Structure):
    _fields_ = [
        ("rgb", k4a_version_t),
        ("depth", k4a_version_t),
        ("audio", k4a_version_t),
        ("depth_sensor", k4a_version_t),
        ("firmware_build", ctypes.c_int),
        ("firmware_signature", ctypes.c_int)
    ]


class k4a_imu_sample_t(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("acc_sample", k4a_float3),
        ("acc_timestamp_usec", ctypes.c_uint64),
        ("gyro_sample", k4a_float3),
        ("gyro_timestamp_usec", ctypes.c_uint64)
    ]
