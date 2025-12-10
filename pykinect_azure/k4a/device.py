import ctypes

from pykinect_azure.k4a import _k4a
from pykinect_azure.k4a._k4atypes import K4A_WAIT_INFINITE
from pykinect_azure.k4a.capture import Capture
from pykinect_azure.k4a.imu_sample import ImuSample
from pykinect_azure.k4a.calibration import Calibration
from pykinect_azure.k4a.transformation import Transformation
from pykinect_azure.k4a.configuration import Configuration
from pykinect_azure.k4arecord.record import Record


class Device:
	def __init__(self, index: int = 0):
		self._handle = self._create_handle(index)
		self.serialnum = self._get_serialnum()
		self.version = self._get_version()
		self.configuration = None
		self.calibration = None
		self.transformation = None
		self.record = None
		self.recording = False

	def __del__(self):
		if self._handle:
			self._stop_imu()
			self._stop_cameras()
			_k4a.k4a_device_close(self._handle)

	def handle(self):
		return self._handle

	def start(
			self, configuration: Configuration, record=False,
			record_filepath="output.mkv"):
		self.configuration = configuration
		self._start_cameras(configuration)
		self._start_imu()
		if record:
			self.record = Record(
				self._handle, self.configuration.handle(), record_filepath)
			self.recording = True

	def update(self, timeout_in_ms: int = K4A_WAIT_INFINITE) -> Capture:
		# Get cameras capture
		capture_handle = self._get_capture(timeout_in_ms)
		capture = Capture(capture_handle, self.transformation)
		if self.recording:
			self.record.write_capture(capture.handle())

		return capture

	def update_imu(self, timeout_in_ms: int = K4A_WAIT_INFINITE) -> ImuSample:
		# Get imu sample
		imu_sample_handle = self._get_imu_sample(timeout_in_ms)
		imu_sample = ImuSample(imu_sample_handle)

		return imu_sample

	@staticmethod
	def device_get_installed_count() -> int:
		return int(_k4a.k4a_device_get_installed_count())

	@staticmethod
	def _create_handle(index: int) -> _k4a.k4a_device_t:
		device_handle = _k4a.k4a_device_t()
		result_code = _k4a.k4a_device_open(index, device_handle)
		if result_code != _k4a.K4A_RESULT_SUCCEEDED:
			raise _k4a.AzureKinectSensorException("Open K4A Device failed")

		return device_handle

	def _start_cameras(self, device_config: Configuration):
		self.calibration = self._get_calibration(
			device_config.depth_mode, device_config.color_resolution)
		self.transformation = Transformation(self.calibration)
		result_code = _k4a.k4a_device_start_cameras(
			self._handle, device_config.handle())
		if result_code != _k4a.K4A_RESULT_SUCCEEDED:
			raise _k4a.AzureKinectSensorException("Start K4A cameras failed.")

	def _stop_cameras(self):
		_k4a.k4a_device_stop_cameras(self._handle)

	def _start_imu(self):
		result_code = _k4a.k4a_device_start_imu(self._handle)
		if result_code != _k4a.K4A_RESULT_SUCCEEDED:
			raise _k4a.AzureKinectSensorException("Start K4A IMU failed.")

	def _stop_imu(self):
		_k4a.k4a_device_stop_imu(self._handle)

	def _get_capture(
			self,
			timeout_in_ms: int = _k4a.K4A_WAIT_INFINITE) -> _k4a.k4a_capture_t:
		capture_handle = _k4a.k4a_capture_t()
		result_code = _k4a.k4a_device_get_capture(
			self._handle, capture_handle, timeout_in_ms)
		if result_code != _k4a.K4A_RESULT_SUCCEEDED:
			raise _k4a.AzureKinectSensorException("Get capture failed.")

		return capture_handle

	def _get_imu_sample(
			self,
			timeout_in_ms: int = _k4a.K4A_WAIT_INFINITE) -> _k4a.k4a_imu_sample_t:
		imu_sample_handle = _k4a.k4a_imu_sample_t()
		result_code = _k4a.k4a_device_get_imu_sample(
			self._handle, imu_sample_handle, timeout_in_ms)
		if result_code != _k4a.K4A_RESULT_SUCCEEDED:
			raise _k4a.AzureKinectSensorException("Get IMU failed.")

		return imu_sample_handle

	def _get_serialnum(self) -> str:
		serial_number_size = ctypes.c_size_t()
		_ = _k4a.k4a_device_get_serialnum(
			self._handle, None, serial_number_size)

		serial_number = ctypes.create_string_buffer(
			serial_number_size.value)
		result_code = _k4a.k4a_device_get_serialnum(
			self._handle, serial_number, serial_number_size)
		if result_code != _k4a.K4A_RESULT_SUCCEEDED:
			raise _k4a.AzureKinectSensorException("Read serial number failed.")

		return serial_number.value.decode("utf-8")

	def _get_calibration(
			self, depth_mode: int, color_resolution: int) -> Calibration:
		calibration_handle = _k4a.k4a_calibration_t()
		result_code = _k4a.k4a_device_get_calibration(
			self._handle, depth_mode, color_resolution, calibration_handle)
		if result_code != _k4a.K4A_RESULT_SUCCEEDED:
			raise _k4a.AzureKinectSensorException("Get calibration failed.")

		return Calibration(calibration_handle)

	def _get_version(self) -> _k4a.k4a_hardware_version_t:
		version = _k4a.k4a_hardware_version_t()
		result_code = _k4a.k4a_device_get_version(self._handle, version)
		if result_code != _k4a.K4A_RESULT_SUCCEEDED:
			raise _k4a.AzureKinectSensorException("Get version failed.")

		return version
