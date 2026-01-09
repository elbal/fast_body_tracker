import ctypes

from ..k4a import k4a_const
from ..k4a import Capture, Calibration, Transformation
from . import _k4abt
from ._k4abt_types import k4abt_frame_t, k4abt_tracker_t
from . import kabt_const
from .tracker_configuration import TrackerConfiguration
from .frame import Frame


class Tracker:
    def __init__(
            self, calibration: Calibration,
            tracker_configuration: TrackerConfiguration):
        self.tracker_configuration = tracker_configuration
        self.calibration = calibration
        self.transformation = Transformation(self.calibration)
        self._handle = self._create_handle()

    def __del__(self):
        if self._handle:
            _k4abt.k4abt_tracker_destroy(self._handle)

    def update(
            self, capture: Capture,
            timeout_in_ms: int = k4a_const.K4A_WAIT_INFINITE) -> Frame:
        result_code = _k4abt.k4abt_tracker_enqueue_capture(
            self._handle, capture.handle(), timeout_in_ms)
        if result_code != kabt_const.K4ABT_RESULT_SUCCEEDED:
            raise _k4abt.AzureKinectBodyTrackerException(
                "Body tracker capture enqueue failed.")

        frame_handle = k4abt_frame_t()
        result_code = _k4abt.k4abt_tracker_pop_result(
            self._handle, ctypes.byref(frame_handle), timeout_in_ms)
        if result_code != kabt_const.K4ABT_RESULT_SUCCEEDED:
            raise _k4abt.AzureKinectBodyTrackerException(
                "Body tracker get body frame failed.")

        return Frame(frame_handle=frame_handle)

    def set_temporal_smoothing(self, smoothing_factor: float):
        _k4abt.k4abt_tracker_set_temporal_smoothing(
            self._handle, smoothing_factor)

    def _create_handle(self) -> k4abt_tracker_t:
        tracker_handle = k4abt_tracker_t()
        result_code = _k4abt.k4abt_tracker_create(
            ctypes.byref(self.calibration.handle()),
            self.tracker_configuration.handle(), ctypes.byref(tracker_handle))
        if result_code != kabt_const.K4ABT_RESULT_SUCCEEDED:
            raise _k4abt.AzureKinectBodyTrackerException(
                "Body tracker initialization failed.")

        return tracker_handle
