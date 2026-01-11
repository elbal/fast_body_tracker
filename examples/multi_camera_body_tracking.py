import queue
import threading
from typing import Tuple

import numpy as np

import fast_body_tracker as fbt


def device_initialization(
        device_index: int = 0) -> Tuple[fbt.Device, fbt.Tracker]:
    device_config = fbt.Configuration()
    device_config.color_format = fbt.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = fbt.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = fbt.K4A_DEPTH_MODE_WFOV_2X2BINNED

    device = fbt.start_device(device_index=device_index, config=device_config)
    tracker = fbt.start_body_tracker(calibration=device.calibration)

    return device, tracker


def main(n_cameras: int = 1):
    fbt.initialize_libraries(track_body=True)

    devices = []
    trackers = []

    stop_event = threading.Event()
    capture_queues = []
    capture_threads = []

    image_array = np.zeros((1080*n_cameras, 1920, 4), dtype=np.uint8)
    image_lock = threading.Lock()
    computation_threads = []

    for i in range(n_cameras):
        device, tracker = device_initialization(device_index=i)
        devices.append(device)
        trackers.append(tracker)

        capture_queues.append(queue.Queue(maxsize=10))
        capture_threads.append(threading.Thread(
            target=fbt.capture_thread,
            args=(devices[i], trackers[i], capture_queues[i], stop_event)))

        computation_threads.append(threading.Thread(
            target=fbt.computation_thread,
            args=(
                i, devices[i].calibration, capture_queues[i], image_array,
                image_lock)))

    for t in computation_threads:
        t.start()
    for t in capture_threads:
        t.start()
    fbt.visualization_main_tread(image_array, image_lock, stop_event)

    for t in capture_threads:
        t.join()
    for t in computation_threads:
        t.join()

    del trackers
    del devices


if __name__ == "__main__":
    main(2)
