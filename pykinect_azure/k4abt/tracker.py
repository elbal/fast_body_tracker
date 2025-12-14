from pykinect_azure.k4a import Capture, Calibration, Transformation
from pykinect_azure.k4abt import _k4abt
from pykinect_azure.k4a._k4atypes import K4A_WAIT_INFINITE
from pykinect_azure.k4abt.trackerconfiguration import TrackerConfiguration
from pykinect_azure.k4abt.frame import Frame


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

	def handle(self):
		return self._handle

	def update(
			self, capture: Capture,
			timeout_in_ms: int = K4A_WAIT_INFINITE) -> Frame:
		result_code = _k4abt.k4abt_tracker_enqueue_capture(
			self._handle, capture.handle(), timeout_in_ms)
		if result_code != _k4abt.K4ABT_RESULT_SUCCEEDED:
			raise _k4abt.AzureKinectBodyTrackerException(
				"Body tracker capture enqueue failed.")

		frame_handle = _k4abt.k4abt_frame_t()
		result_code = _k4abt.k4abt_tracker_pop_result(
			self._handle, frame_handle, timeout_in_ms)
		if result_code != _k4abt.K4ABT_RESULT_SUCCEEDED:
			raise _k4abt.AzureKinectBodyTrackerException(
				"Body tracker get body frame failed.")

		return Frame(
			frame_handle=frame_handle, calibration=self.calibration,
			transformation=self.transformation)

	def set_temporal_smoothing(self, smoothing_factor: float):
		_k4abt.k4abt_tracker_set_temporal_smoothing(
			self._handle, smoothing_factor)

	def _create_handle(self) -> _k4abt.k4abt_tracker_t:
		tracker_handle = _k4abt.k4abt_tracker_t()
		result_code = _k4abt.k4abt_tracker_create(
			self.calibration.handle(), self.tracker_configuration.handle(),
			tracker_handle)
		if result_code != _k4abt.K4ABT_RESULT_SUCCEEDED:
			raise _k4abt.AzureKinectBodyTrackerException(
				"Body tracker initialization failed.")

		return tracker_handle
