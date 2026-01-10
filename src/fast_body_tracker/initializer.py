import platform
from pathlib import Path
import os

from .k4a._k4a import K4aLib
from .k4a.device import Device
from .k4a.configuration import Configuration
from .k4abt import _k4abt, Tracker, TrackerConfiguration
from .k4arecord import _k4arecord
from .k4arecord.playback import Playback


class SDKNotImplemented(Exception):
    """
    Raised when the SDK is not supported by the platform at hand.
    """


def initialize_libraries(
        module_k4a_path=None, module_k4abt_path=None, track_body=False):
    if module_k4a_path is None:
        module_k4a_path = _get_k4a_module_path()
    K4aLib.setup(module_k4a_path)

    if track_body:
        if module_k4abt_path is None:
            module_k4abt_path = _get_k4abt_module_path()
        _k4abt.setup_library(module_k4abt_path)

    module_k4arecord_path = _get_k4arecord_module_path(module_k4a_path)
    _k4arecord.setup_library(module_k4arecord_path)


def start_device(
        device_index=0, config=Configuration(), record=False,
        record_filepath="output.mkv"):
    device = Device(device_index)
    device.start(config, record, record_filepath)

    return device


def start_body_tracker(
        calibration, tracker_configuration=TrackerConfiguration()):
    return Tracker(calibration, tracker_configuration)


def start_playback(filepath):
    return Playback(filepath)


def _get_k4a_module_path():
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        if machine == "aarch64":
            return "/usr/lib/aarch64-linux-gnu/libk4a.so"
        return "/usr/lib/x86_64-linux-gnu/libk4a.so"

    if system == "windows":
        base = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        arch_folder = "amd64" if machine == "amd64" else "x86"
        for version in ["v1.4.2", "v1.4.1"]:
            dll_path = (
                    base / f"Azure Kinect SDK {version}" / "sdk"
                    / "windows-desktop" / arch_folder / "release" / "bin"
                    / "k4a.dll")
            if dll_path.exists():
                return str(dll_path)
        raise FileNotFoundError(
            "Compatible Azure Kinect SDK (v1.4.1 or v1.4.2) "
            "not found in Program Files.")

    raise OSError(f"Unsupported operating system: {system}")


def _get_k4abt_module_path():
    system = platform.system().lower()

    if platform.machine().lower() == "aarch64":
        raise SDKNotImplemented("ARM is not supported")

    if system == "linux":
        return "libk4abt.so"

    if system == "windows":
        base = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        full_path = (
                base / "Azure Kinect Body Tracking SDK" / "sdk"
                / "windows-desktop" / "amd64" / "release" / "bin"
                / "k4abt.dll")
        if full_path.exists():
            return str(full_path)
        return "k4abt.dll"

    raise OSError(f"Unsupported operating system: {system}")


def _get_k4arecord_module_path(module_path):
    return module_path.replace("k4a", "k4arecord")
