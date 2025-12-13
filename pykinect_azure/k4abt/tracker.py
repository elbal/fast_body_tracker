import platform
from pathlib import Path
import os

from pykinect_azure.k4a.calibration import Calibration
from pykinect_azure.k4a._k4atypes import K4A_WAIT_INFINITE
from pykinect_azure.k4abt.trackerconfiguration import TrackerConfiguration
from pykinect_azure.k4abt import _k4abt
from pykinect_azure.k4abt.frame import Frame
from pykinect_azure.k4abt._k4abtTypes import (
	k4abt_tracker_default_configuration)


class Tracker:
	def __init__(
			self, calibration: Calibration, model_type,
			tracker_configuration: TrackerConfiguration):
		self.tracker_configuration = tracker_configuration
		self.calibration = calibration
		self._handle = self._create_handle(model_type)

	def __del__(self):
		if self._handle:
			_k4abt.k4abt_tracker_destroy(self._handle)

	def handle(self):
		return self._handle

	def update(self, capture, timeout_in_ms=K4A_WAIT_INFINITE):
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

		return Frame(frame_handle, self.calibration)

	def set_temporal_smoothing(self, smoothing_factor):
		_k4abt.k4abt_tracker_set_temporal_smoothing(
			self._handle, smoothing_factor)

	def get_tracker_configuration(self, model_type):
		tracker_config = k4abt_tracker_default_configuration

		if model_type == _k4abt.K4ABT_LITE_MODEL:
			tracker_config.model_path = self._get_k4abt_lite_model_path()

		return tracker_config

	def _create_handle(self, model_type):
		if model_type == _k4abt.K4ABT_LITE_MODEL:
			self.tracker_configuration.model_path = (
				self._get_k4abt_lite_model_path())

		tracker_handle = _k4abt.k4abt_tracker_t()
		result_code = _k4abt.k4abt_tracker_create(
			self.calibration.handle(), self.tracker_configuration.handle(),
			tracker_handle)
		if result_code != _k4abt.K4ABT_RESULT_SUCCEEDED:
			raise _k4abt.AzureKinectBodyTrackerException(
				"Body tracker initialization failed.")

		return tracker_handle

	@staticmethod
	def _get_k4abt_lite_model_path():
		system = platform.system().lower()

		if system == "linux":
			raise OSError(f"Unsupported operating system: {system}")

		if system == "windows":
			base = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
			full_path = (
					base / "Azure Kinect Body Tracking SDK" / "sdk"
					/ "windows-desktop" / "amd64" / "release" / "bin"
					/ "dnn_model_2_0_lite_op11.onnx")
			return str(full_path).encode('utf-8')

		raise OSError(f"Unsupported operating system: {system}")
