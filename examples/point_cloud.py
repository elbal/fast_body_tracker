import threading
import queue
import fast_body_tracker as fbt
from fast_body_tracker import PointCloudVisualizer, KeyboardCloser


def capture_thread(device, q, stop_event):
    dfa = fbt.DroppedFramesAlert()
    while not stop_event.is_set():
        capture = device.update()
        if q.full():
            dfa.update()
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put(capture)


def main():
    fbt.initialize_libraries()

    device_config = fbt.Configuration()
    device_config.color_resolution = fbt.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = fbt.K4A_DEPTH_MODE_WFOV_2X2BINNED

    device = fbt.start_device(config=device_config)
    transformation = device.transformation
    q = queue.Queue(maxsize=30)
    keyboard_closer = KeyboardCloser()
    keyboard_closer.start()
    t = threading.Thread(
        target=capture_thread, args=(device, q, keyboard_closer.stop_event))

    visualizer = PointCloudVisualizer()
    point_cloud_object = None

    t.start()
    while not keyboard_closer.stop_event.is_set():
        capture = q.get()
        depth_image_object = capture.get_depth_image_object()
        point_cloud_object = transformation.depth_image_to_point_cloud(
            depth_image_object, point_cloud_object,
            calibration_type=fbt.K4A_CALIBRATION_TYPE_DEPTH)

        point_cloud = point_cloud_object.to_numpy()
        visualizer.update(point_cloud)

    t.join()
    del device


if __name__ == "__main__":
    main()
