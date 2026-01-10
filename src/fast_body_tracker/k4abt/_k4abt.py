import ctypes
import platform
import sys

from ..k4a import _k4a_types
from . import _k4abt_types
from . import kabt_const

k4abt_dll = None

k4abt_tracker_default_configuration = (
    _k4abt_types.k4abt_tracker_configuration_t())
k4abt_tracker_default_configuration.sensor_orientation = (
    kabt_const.K4ABT_SENSOR_ORIENTATION_DEFAULT)
k4abt_tracker_default_configuration.processing_mode = (
    kabt_const.K4ABT_TRACKER_PROCESSING_MODE_GPU)
k4abt_tracker_default_configuration.gpu_device_id = 0


def setup_library(module_k4abt_path):
    global k4abt_dll
    try:
        k4abt_dll = ctypes.CDLL(module_k4abt_path)
    except Exception as e:
        print("Failed to load body tracker library", e)
        sys.exit(1)
    setup_onnx_provider()


def setup_onnx_provider():
    if platform.system() == "Windows":
        setup_onnx_provider_windows()
    elif platform.system() == "Linux":
        setup_onnx_provider_linux()


def setup_onnx_provider_linux():
    _k4abt_types.k4abt_tracker_default_configuration.processing_mode = (
        kabt_const.K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA)
    try:
        ctypes.cdll.LoadLibrary("libonnxruntime_providers_cuda.so")
    except Exception as e:
        ctypes.cdll.LoadLibrary("libonnxruntime.so.1.10.0")
        _k4abt_types.k4abt_tracker_default_configuration.processing_mode = (
            kabt_const.K4ABT_TRACKER_PROCESSING_MODE_CPU)


def setup_onnx_provider_windows():
    try:
        ctypes.cdll.LoadLibrary(
            "C:/Program Files/Azure Kinect Body Tracking SDK/tools/"
            "directml.dll")
    except Exception as e:
        try:
            ctypes.cdll.LoadLibrary(
                "C:/Program Files/Azure Kinect Body Tracking SDK/sdk/"
                "windows-desktop/amd64/release/bin/"
                "onnxruntime_providers_cuda.dll")
            _k4abt_types.k4abt_tracker_default_configuration.processing_mode = (
                kabt_const.K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA)
        except Exception as e:
            _k4abt_types.k4abt_tracker_default_configuration.processing_mode = (
                kabt_const.K4ABT_TRACKER_PROCESSING_MODE_CPU)


def k4abt_tracker_create(sensor_calibration, config, tracker_handle):
    _k4abt_tracker_create = k4abt_dll.k4abt_tracker_create
    _k4abt_tracker_create.restype = ctypes.c_int
    _k4abt_tracker_create.argtypes = (
        ctypes.POINTER(_k4a_types.k4a_calibration_t),
        _k4abt_types.k4abt_tracker_configuration_t,
        ctypes.POINTER(_k4abt_types.k4abt_tracker_t))

    return _k4abt_tracker_create(sensor_calibration, config, tracker_handle)


def k4abt_tracker_destroy(tracker_handle):
    _k4abt_tracker_destroy = k4abt_dll.k4abt_tracker_destroy
    _k4abt_tracker_destroy.argtypes = (_k4abt_types.k4abt_tracker_t,)

    _k4abt_tracker_destroy(tracker_handle)


def k4abt_tracker_set_temporal_smoothing(tracker_handle, smoothing_factor):
    _k4abt_tracker_set_temporal_smoothing = (
        k4abt_dll.k4abt_tracker_set_temporal_smoothing)
    _k4abt_tracker_set_temporal_smoothing.argtypes = (
        _k4abt_types.k4abt_tracker_t, ctypes.c_float)

    _k4abt_tracker_set_temporal_smoothing(tracker_handle, smoothing_factor)


def k4abt_tracker_enqueue_capture(
        tracker_handle, sensor_capture_handle, timeout_in_ms):
    _k4abt_tracker_enqueue_capture = k4abt_dll.k4abt_tracker_enqueue_capture
    _k4abt_tracker_enqueue_capture.restype = ctypes.c_int
    _k4abt_tracker_enqueue_capture.argtypes = (
        _k4abt_types.k4abt_tracker_t, _k4a_types.k4a_capture_t, ctypes.c_int32)

    return _k4abt_tracker_enqueue_capture(
        tracker_handle, sensor_capture_handle, timeout_in_ms)


def k4abt_tracker_pop_result(tracker_handle, body_frame_handle, timeout_in_ms):
    _k4abt_tracker_pop_result = k4abt_dll.k4abt_tracker_pop_result
    _k4abt_tracker_pop_result.restype = ctypes.c_int
    _k4abt_tracker_pop_result.argtypes = (
        _k4abt_types.k4abt_tracker_t,
        ctypes.POINTER(_k4abt_types.k4abt_frame_t), ctypes.c_int32)

    return _k4abt_tracker_pop_result(
        tracker_handle, body_frame_handle, timeout_in_ms)


def k4abt_tracker_shutdown(tracker_handle):
    _k4abt_tracker_shutdown = k4abt_dll.k4abt_tracker_shutdown
    _k4abt_tracker_shutdown.argtypes = (_k4abt_types.k4abt_tracker_t,)

    _k4abt_tracker_shutdown(tracker_handle)


def k4abt_frame_release(body_frame_handle):
    _k4abt_frame_release = k4abt_dll.k4abt_frame_release
    _k4abt_frame_release.argtypes = (_k4abt_types.k4abt_frame_t,)

    _k4abt_frame_release(body_frame_handle)


def k4abt_frame_reference(body_frame_handle):
    _k4abt_frame_reference = k4abt_dll.k4abt_frame_reference
    _k4abt_frame_reference.argtypes = (_k4abt_types.k4abt_frame_t,)

    _k4abt_frame_reference(body_frame_handle)


def k4abt_frame_get_num_bodies(body_frame_handle):
    _k4abt_frame_get_num_bodies = k4abt_dll.k4abt_frame_get_num_bodies
    _k4abt_frame_get_num_bodies.restype = ctypes.c_uint32
    _k4abt_frame_get_num_bodies.argtypes = (_k4abt_types.k4abt_frame_t,)

    return _k4abt_frame_get_num_bodies(body_frame_handle)


def k4abt_frame_get_body_skeleton(body_frame_handle, index, skeleton):
    _k4abt_frame_get_body_skeleton = k4abt_dll.k4abt_frame_get_body_skeleton
    _k4abt_frame_get_body_skeleton.restype = ctypes.c_int
    _k4abt_frame_get_body_skeleton.argtypes = (
        _k4abt_types.k4abt_frame_t, ctypes.c_uint32,
        ctypes.POINTER(_k4abt_types.k4abt_skeleton_t))

    return _k4abt_frame_get_body_skeleton(body_frame_handle, index, skeleton)


def k4abt_frame_get_body_id(body_frame_handle, index):
    _k4abt_frame_get_body_id = k4abt_dll.k4abt_frame_get_body_id
    _k4abt_frame_get_body_id.restype = ctypes.c_uint32
    _k4abt_frame_get_body_id.argtypes = (
        _k4abt_types.k4abt_frame_t, ctypes.c_uint32)

    return _k4abt_frame_get_body_id(body_frame_handle, index)


def k4abt_frame_get_device_timestamp_usec(body_frame_handle):
    _k4abt_frame_get_device_timestamp_usec = (
        k4abt_dll.k4abt_frame_get_device_timestamp_usec)
    _k4abt_frame_get_device_timestamp_usec.restype = ctypes.c_uint64
    _k4abt_frame_get_device_timestamp_usec.argtypes = (
        _k4abt_types.k4abt_frame_t,)

    return _k4abt_frame_get_device_timestamp_usec(body_frame_handle)


def k4abt_frame_get_body_index_map(body_frame_handle):
    _k4abt_frame_get_body_index_map = (
        k4abt_dll.k4abt_frame_get_body_index_map)
    _k4abt_frame_get_body_index_map.restype = _k4a_types.k4a_image_t
    _k4abt_frame_get_body_index_map.argtypes = (_k4abt_types.k4abt_frame_t,)

    return _k4abt_frame_get_body_index_map(body_frame_handle)


def k4abt_frame_get_capture(body_frame_handle):
    _k4abt_frame_get_capture = k4abt_dll.k4abt_frame_get_capture
    _k4abt_frame_get_capture.restype = _k4a_types.k4a_capture_t
    _k4abt_frame_get_capture.argtypes = (_k4abt_types.k4abt_frame_t,)

    return _k4abt_frame_get_capture(body_frame_handle)


class AzureKinectBodyTrackerException(Exception):
    pass
