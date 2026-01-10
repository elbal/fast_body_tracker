import threading
import queue
import fast_body_tracker as pykinect

from fast_body_tracker import PointCloudVisualizer, KeyboardCloser


def capture_thread(device, q, stop_event):
    dfa = pykinect.DroppedFramesAlert()
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
    pykinect.initialize_libraries()

    device_config = pykinect.Configuration()
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device_config.synchronized_images_only = True

    device = pykinect.start_device(config=device_config)
    transformation = device.transformation
    q = queue.Queue(maxsize=30)
    keyboard_closer = KeyboardCloser()
    keyboard_closer.start()
    t = threading.Thread(
        target=capture_thread, args=(device, q, keyboard_closer.stop_event))

    visualizer = PointCloudVisualizer()
    point_cloud_object = None
    transformed_image_object = None

    t.start()
    while not keyboard_closer.stop_event.is_set():
        capture = q.get()

        depth_image_object = capture.get_depth_image_object()
        point_cloud_object = transformation.depth_image_to_point_cloud(
            depth_image_object, point_cloud_object)
        point_cloud = point_cloud_object.to_numpy()

        color_image_object = capture.get_color_image_object()
        transformed_image_object = transformation.color_image_to_depth_camera(
            depth_image_object, color_image_object, transformed_image_object)
        bgra_image = transformed_image_object.to_numpy()

        visualizer.update(point_cloud, bgra_image)

    t.join()
    del device


if __name__ == "__main__":
    main()
