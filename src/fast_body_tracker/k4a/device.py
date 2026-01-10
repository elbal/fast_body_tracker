import ctypes

from ._k4a_types import (
    k4a_calibration_t, k4a_capture_t, k4a_device_t, k4a_hardware_version_t,
    k4a_imu_sample_t)
from . import _k4a
from . import k4a_const
from .capture import Capture
from .calibration import Calibration
from .configuration import Configuration
from .imu_sample import ImuSample
from .transformation import Transformation
from ..k4arecord.record import Record


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
            _k4a.K4aLib.k4a_device_close(self._handle)

    def handle(self):
        return self._handle

    def start(
            self, configuration: Configuration, record=False,
            record_filepath="output.mkv"):
        self.configuration = configuration
        self.calibration = self._get_calibration(
            configuration.depth_mode, configuration.color_resolution)
        self.transformation = Transformation(self.calibration)
        self._start_cameras(configuration)
        self._start_imu()
        if record:
            self.record = Record(
                self._handle, self.configuration.handle(), record_filepath)
            self.recording = True

    def update(self, timeout_in_ms: int = k4a_const.K4A_WAIT_INFINITE) -> (
            Capture):
        capture_handle = self._get_capture(timeout_in_ms)
        capture = Capture(capture_handle)
        if self.recording:
            self.record.write_capture(capture.handle())

        return capture

    def update_imu(
            self,
            timeout_in_ms: int = k4a_const.K4A_WAIT_INFINITE) -> ImuSample:
        imu_sample_handle = self._get_imu_sample(timeout_in_ms)
        imu_sample = ImuSample(imu_sample_handle)

        return imu_sample

    @staticmethod
    def device_get_installed_count() -> int:
        return int(_k4a.K4aLib.k4a_device_get_installed_count())

    @staticmethod
    def _create_handle(index: int) -> k4a_device_t:
        device_handle = k4a_device_t()
        result_code = _k4a.K4aLib.k4a_device_open(
            index, ctypes.byref(device_handle))
        if result_code != k4a_const.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException("Open K4A Device failed")

        return device_handle

    def _start_cameras(self, configuration: Configuration):
        result_code = _k4a.K4aLib.k4a_device_start_cameras(
            self._handle, ctypes.byref(configuration.handle()))
        if result_code != k4a_const.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException("Start K4A cameras failed.")

    def _stop_cameras(self):
        _k4a.K4aLib.k4a_device_stop_cameras(self._handle)

    def _start_imu(self):
        result_code = _k4a.K4aLib.k4a_device_start_imu(self._handle)
        if result_code != k4a_const.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException("Start K4A IMU failed.")

    def _stop_imu(self):
        _k4a.K4aLib.k4a_device_stop_imu(self._handle)

    def _get_capture(
            self,
            timeout_in_ms: int = k4a_const.K4A_WAIT_INFINITE) -> k4a_capture_t:
        capture_handle = k4a_capture_t()
        result_code = _k4a.K4aLib.k4a_device_get_capture(
            self._handle, ctypes.byref(capture_handle), timeout_in_ms)
        if result_code != k4a_const.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException("Get capture failed.")

        return capture_handle

    def _get_imu_sample(
            self,
            timeout_in_ms: int = k4a_const.K4A_WAIT_INFINITE) -> k4a_imu_sample_t:
        imu_sample_handle = k4a_imu_sample_t()
        result_code = _k4a.K4aLib.k4a_device_get_imu_sample(
            self._handle, ctypes.byref(imu_sample_handle), timeout_in_ms)
        if result_code != k4a_const.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException("Get IMU failed.")

        return imu_sample_handle

    def _get_serialnum(self) -> str:
        serial_number_size = ctypes.c_size_t()
        _ = _k4a.K4aLib.k4a_device_get_serialnum(
            self._handle, None, ctypes.byref(serial_number_size))

        serial_number = ctypes.create_string_buffer(
            serial_number_size.value)
        result_code = _k4a.K4aLib.k4a_device_get_serialnum(
            self._handle, serial_number, ctypes.byref(serial_number_size))
        if result_code != _k4a.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException("Read serial number failed.")

        return serial_number.value.decode("utf-8")

    def _get_calibration(
            self, depth_mode: int, color_resolution: int) -> Calibration:
        calibration_handle = k4a_calibration_t()
        result_code = _k4a.K4aLib.k4a_device_get_calibration(
            self._handle, depth_mode, color_resolution,
            ctypes.byref(calibration_handle))
        if result_code != _k4a.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException("Get calibration failed.")

        return Calibration(calibration_handle)

    def _get_version(self) -> k4a_hardware_version_t:
        version = k4a_hardware_version_t()
        result_code = _k4a.K4aLib.k4a_device_get_version(
            self._handle, ctypes.byref(version))
        if result_code != k4a_const.K4A_RESULT_SUCCEEDED:
            raise _k4a.AzureKinectSensorException("Get version failed.")

        return version
