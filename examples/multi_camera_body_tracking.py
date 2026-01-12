import queue
import threading
import pathlib
from datetime import datetime
from typing import Tuple

import fast_body_tracker as fbt


def device_initialization(
        device_index: int = 0,
        device_mode: str = "standalone") -> Tuple[fbt.Device, fbt.Tracker]:
    modes = {
        "standalone": fbt.K4A_WIRED_SYNC_MODE_STANDALONE,
        "main": fbt.K4A_WIRED_SYNC_MODE_MASTER,
        "secondary": fbt.K4A_WIRED_SYNC_MODE_SUBORDINATE}

    device_config = fbt.Configuration()
    device_config.color_format = fbt.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = fbt.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = fbt.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device_config.synchronized_images_only = True
    device_config.wired_sync_mode = modes[device_mode]

    device = fbt.start_device(device_index=device_index, config=device_config)
    tracker = fbt.start_body_tracker(calibration=device.calibration)

    return device, tracker


def main(n_devices: int = 1):
    fbt.initialize_libraries(track_body=True)

    devices = []
    trackers = []

    capture_queues = []
    capture_threads = []
    stop_event = threading.Event()

    computation_threads = []
    joints_queue = queue.Queue(maxsize=10)
    video_queue = queue.Queue(maxsize=10)
    visualization_queue = queue.Queue(maxsize=10)

    for i in range(n_devices):
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
                i, devices[i].calibration, capture_queues[i], joints_queue,
                video_queue, visualization_queue)))

    base_dir = pathlib.Path("../data/")
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    file_dir = base_dir / timestamp
    file_dir.mkdir(parents=True, exist_ok=True)
    joints_saver_thread = threading.Thread(
        target=fbt.joints_saver_thread,
        args=(n_devices, joints_queue, file_dir))
    video_saver_thread = threading.Thread(
        target=fbt.video_saver_thread, args=(n_devices, video_queue, file_dir))

    video_saver_thread.start()
    joints_saver_thread.start()
    for t in computation_threads:
        t.start()
    for t in capture_threads:
        t.start()
    fbt.visualization_main_tread(n_devices, visualization_queue, stop_event)

    video_saver_thread.join()
    joints_saver_thread.join()
    for t in capture_threads:
        t.join()
    for t in computation_threads:
        t.join()
    del trackers
    del devices


if __name__ == "__main__":
    main(2)
