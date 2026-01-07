import platform
from pathlib import Path
import os

from ._k4abt_types import k4abt_tracker_configuration_t
from . import kabt_const


class UnknownModelType(Exception):
    """
    Exception raised when trying to use an unknown model.
    """


class TrackerConfiguration:
    def __init__(self):
        self.sensor_orientation = kabt_const.K4ABT_SENSOR_ORIENTATION_DEFAULT
        self.tracker_processing_mode = (
            kabt_const.K4ABT_TRACKER_PROCESSING_MODE_GPU)
        self.gpu_device_id = 0
        self.model_type = kabt_const.K4ABT_DEFAULT_MODEL

        self._handle = self._on_value_change()

    def handle(self):
        return self._handle

    def __setattr__(self, name: str, value: int):
        if hasattr(self, name):
            if name != "_handle":
                if self.__dict__[name] != value:
                    self.__dict__[name] = value
                    self._handle = self._on_value_change()
            else:
                self.__dict__[name] = value
        else:
            self.__dict__[name] = value

    def __str__(self):
        message = (
            "Device configuration: \n"
            f"\tsensor_orientation: {self.sensor_orientation} "
            f"\n\t(0: Default, 1: Clockwise90, 2: CounterClockwise90, "
            f"3: Flip180)\n\n"
            f"\ttracker_processing_mode: {self.tracker_processing_mode} "
            f"\n\t(0:Gpu, 1:Cpu, 2:CUDA, 3:TensorRT, 4:DirectML)\n\n"
            f"\tgpu_device_id: {self.gpu_device_id}\n\n"
            f"\tmodel_path: {
            self.model_path if hasattr(self, 'model_path')
            else 'Default Model'}")

        return message

    def _on_value_change(self):
        if self.model_type == kabt_const.K4ABT_DEFAULT_MODEL:
            configuration_handle = k4abt_tracker_configuration_t(
                self.sensor_orientation, self.tracker_processing_mode,
                self.gpu_device_id)
        elif self.model_type == kabt_const.K4ABT_LITE_MODEL:
            model_path = self._get_k4abt_lite_model_path()
            configuration_handle = k4abt_tracker_configuration_t(
                self.sensor_orientation, self.tracker_processing_mode,
                self.gpu_device_id, model_path)
        else:
            raise UnknownModelType("Unknown model type.")

        return configuration_handle

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
