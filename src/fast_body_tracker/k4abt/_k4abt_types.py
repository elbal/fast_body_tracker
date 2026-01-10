import ctypes

from ..k4a._k4a_types import k4a_float3
from . import kabt_const

k4abt_result_t = ctypes.c_int
k4abt_float4 = ctypes.c_float * 4


class _handle_k4abt_tracker_t(ctypes.Structure):
    _fields_ = [("_rsvd", ctypes.c_size_t)]


k4abt_tracker_t = ctypes.POINTER(_handle_k4abt_tracker_t)


class _handle_k4abt_frame_t(ctypes.Structure):
    _fields_ = [("_rsvd", ctypes.c_size_t)]


k4abt_frame_t = ctypes.POINTER(_handle_k4abt_frame_t)


class k4abt_tracker_configuration_t(ctypes.Structure):
    _fields_ = [
        ("sensor_orientation", ctypes.c_int),
        ("processing_mode", ctypes.c_int),
        ("gpu_device_id", ctypes.c_int32),
        ("model_path", ctypes.c_char_p)
    ]


class k4abt_joint_t(ctypes.Structure):
    _fields_ = [
        ("position", k4a_float3),
        ("orientation", k4abt_float4),
        ("confidence_level", ctypes.c_int)]


class k4abt_skeleton_t(ctypes.Structure):
    _fields_ = [("joints", k4abt_joint_t * kabt_const.K4ABT_JOINT_COUNT)]


class k4abt_body_t(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("skeleton", k4abt_skeleton_t)]
