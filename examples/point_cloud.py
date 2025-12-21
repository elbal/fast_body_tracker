import threading
import queue
import pykinect_azure as pykinect
from pykinect_azure import PointCloudVisualizer, KeyboardCloser


def capture_thread(device, q, stop_event):
    while not stop_event.is_set():
        capture = device.update()
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put(capture)


def main():
    pykinect.initialize_libraries()

    device_config = pykinect.Configuration()
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

    device = pykinect.start_device(config=device_config)
    transformation = device.transformation

    q = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    t = threading.Thread(target=capture_thread, args=(device, q, stop_event))
    t.start()

    visualizer = PointCloudVisualizer()
    keyboard_closer = KeyboardCloser()
    keyboard_closer.start()

    point_cloud_object = None

    while not keyboard_closer.stop_event.is_set():
        capture = q.get()
        depth_image_object = capture.get_depth_image_object()
        point_cloud_object = transformation.depth_image_to_point_cloud(
            depth_image_object, point_cloud_object,
            calibration_type=pykinect.K4A_CALIBRATION_TYPE_DEPTH)

        point_cloud = point_cloud_object.to_numpy()
        visualizer(point_cloud)

    stop_event.set()
    t.join()
    del device


if __name__ == "__main__":
    main()
