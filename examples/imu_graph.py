import threading
import queue

import pykinect_azure as pykinect
from pykinect_azure import IMUVisualizer, KeyboardCloser


def capture_thread(device, q, stop_event):
    while not stop_event.is_set():
        dfa = pykinect.DroppedFramesAlert()
        imu_sample = device.update_imu()
        if q.full():
            dfa.update()
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put(imu_sample)


def main():
    pykinect.initialize_libraries()

    device_config = pykinect.Configuration()
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF

    device = pykinect.start_device(config=device_config)
    q = queue.Queue(maxsize=200)
    keyboard_closer = KeyboardCloser()
    keyboard_closer.start()
    t = threading.Thread(
        target=capture_thread, args=(device, q, keyboard_closer.stop_event))

    visualizer = IMUVisualizer()

    t.start()
    while not keyboard_closer.stop_event.is_set():
        samples = [q.get()]
        while not q.empty():
            samples.append(q.get_nowait())
        visualizer.update(samples)

    t.join()
    del device


if __name__ == "__main__":
    main()
