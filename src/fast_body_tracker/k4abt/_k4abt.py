import ctypes
import platform

from ..k4a import _k4a_types
from . import _k4abt_types
from . import kabt_const


class AzureKinectBodyTrackerException(Exception):
    pass


class K4abLib:
    _dll = None
    _default_tracker_processing_mode = kabt_const.K4ABT_TRACKER_PROCESSING_MODE_GPU

    k4abt_tracker_create = None
    k4abt_tracker_destroy = None
    k4abt_tracker_set_temporal_smoothing = None
    k4abt_tracker_enqueue_capture = None
    k4abt_tracker_pop_result = None
    k4abt_tracker_shutdown = None

    k4abt_frame_release = None
    k4abt_frame_reference = None
    k4abt_frame_get_num_bodies = None
    k4abt_frame_get_body_skeleton = None
    k4abt_frame_get_body_id = None
    k4abt_frame_get_device_timestamp_usec = None
    k4abt_frame_get_body_index_map = None
    k4abt_frame_get_capture = None

    @classmethod
    def setup(cls, path):
        if cls._dll is not None:
            return

        try:
            cls._dll = ctypes.CDLL(path)
        except Exception as exc:
            raise AzureKinectBodyTrackerException(
                "Failed to load body tracker library."
            ) from exc

        cls._setup_onnx_provider()
        cls._bind_all()

    @classmethod
    def default_tracker_configuration(cls):
        configuration = _k4abt_types.k4abt_tracker_configuration_t()
        configuration.sensor_orientation = kabt_const.K4ABT_SENSOR_ORIENTATION_DEFAULT
        configuration.processing_mode = cls._default_tracker_processing_mode
        configuration.gpu_device_id = 0
        configuration.model_path = None
        return configuration

    @classmethod
    def default_tracker_processing_mode(cls):
        return cls._default_tracker_processing_mode

    @classmethod
    def _bind(cls, name, restype, argtypes):
        func = getattr(cls._dll, name)
        func.restype = restype
        func.argtypes = argtypes
        setattr(cls, name, func)

    @classmethod
    def _bind_all(cls):
        cls._bind(
            "k4abt_tracker_create",
            ctypes.c_int,
            (
                ctypes.POINTER(_k4a_types.k4a_calibration_t),
                _k4abt_types.k4abt_tracker_configuration_t,
                ctypes.POINTER(_k4abt_types.k4abt_tracker_t),
            ),
        )
        cls._bind("k4abt_tracker_destroy", None, (_k4abt_types.k4abt_tracker_t,))
        cls._bind(
            "k4abt_tracker_set_temporal_smoothing",
            None,
            (_k4abt_types.k4abt_tracker_t, ctypes.c_float),
        )
        cls._bind(
            "k4abt_tracker_enqueue_capture",
            ctypes.c_int,
            (
                _k4abt_types.k4abt_tracker_t,
                _k4a_types.k4a_capture_t,
                ctypes.c_int32,
            ),
        )
        cls._bind(
            "k4abt_tracker_pop_result",
            ctypes.c_int,
            (
                _k4abt_types.k4abt_tracker_t,
                ctypes.POINTER(_k4abt_types.k4abt_frame_t),
                ctypes.c_int32,
            ),
        )
        cls._bind("k4abt_tracker_shutdown", None, (_k4abt_types.k4abt_tracker_t,))

        cls._bind("k4abt_frame_release", None, (_k4abt_types.k4abt_frame_t,))
        cls._bind("k4abt_frame_reference", None, (_k4abt_types.k4abt_frame_t,))
        cls._bind(
            "k4abt_frame_get_num_bodies",
            ctypes.c_uint32,
            (_k4abt_types.k4abt_frame_t,),
        )
        cls._bind(
            "k4abt_frame_get_body_skeleton",
            ctypes.c_int,
            (
                _k4abt_types.k4abt_frame_t,
                ctypes.c_uint32,
                ctypes.POINTER(_k4abt_types.k4abt_skeleton_t),
            ),
        )
        cls._bind(
            "k4abt_frame_get_body_id",
            ctypes.c_uint32,
            (_k4abt_types.k4abt_frame_t, ctypes.c_uint32),
        )
        cls._bind(
            "k4abt_frame_get_device_timestamp_usec",
            ctypes.c_uint64,
            (_k4abt_types.k4abt_frame_t,),
        )
        cls._bind(
            "k4abt_frame_get_body_index_map",
            _k4a_types.k4a_image_t,
            (_k4abt_types.k4abt_frame_t,),
        )
        cls._bind(
            "k4abt_frame_get_capture",
            _k4a_types.k4a_capture_t,
            (_k4abt_types.k4abt_frame_t,),
        )

    @classmethod
    def _setup_onnx_provider(cls):
        system = platform.system()
        if system == "Windows":
            cls._setup_onnx_provider_windows()
        elif system == "Linux":
            cls._setup_onnx_provider_linux()

    @classmethod
    def _setup_onnx_provider_linux(cls):
        cls._default_tracker_processing_mode = (
            kabt_const.K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA
        )
        try:
            ctypes.cdll.LoadLibrary("libonnxruntime_providers_cuda.so")
        except Exception:
            ctypes.cdll.LoadLibrary("libonnxruntime.so.1.10.0")
            cls._default_tracker_processing_mode = (
                kabt_const.K4ABT_TRACKER_PROCESSING_MODE_CPU
            )

    @classmethod
    def _setup_onnx_provider_windows(cls):
        try:
            ctypes.cdll.LoadLibrary(
                "C:/Program Files/Azure Kinect Body Tracking SDK/sdk/"
                "windows-desktop/amd64/release/bin/"
                "onnxruntime_providers_cuda.dll"
            )
            cls._default_tracker_processing_mode = (
                kabt_const.K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA
            )
        except Exception:
            try:
                ctypes.cdll.LoadLibrary(
                    "C:/Program Files/Azure Kinect Body Tracking SDK/tools/directml.dll"
                )
                cls._default_tracker_processing_mode = (
                    kabt_const.K4ABT_TRACKER_PROCESSING_MODE_GPU
                )
            except Exception:
                cls._default_tracker_processing_mode = (
                    kabt_const.K4ABT_TRACKER_PROCESSING_MODE_CPU
                )
