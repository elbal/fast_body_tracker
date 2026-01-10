import cv2
import numpy as np
import threading
import queue
import fast_body_tracker as pykinect


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
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    device_config.synchronized_images_only = True

    device = pykinect.start_device(config=device_config)
    transformation = device.transformation
    q = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    t = threading.Thread(target=capture_thread, args=(device, q, stop_event))

    cv2.namedWindow("Transformed depth image", cv2.WINDOW_NORMAL)
    transformed_image_object = None
    gray_image = np.zeros((1080, 1920), dtype=np.uint8)
    gray_3channel_image = np.empty((1080, 1920, 3), dtype=np.uint8)
    depth_8bit_image = np.zeros((1080, 1920), dtype=np.uint8)
    depth_colorized_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    combined_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    frc = pykinect.FrameRateCalculator()

    t.start()
    frc.start()
    while True:
        capture = q.get()

        color_image_object = capture.get_color_image_object()
        bgra_image = color_image_object.to_numpy()
        cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2GRAY, dst=gray_image)
        cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR, dst=gray_3channel_image)

        depth_image_object = capture.get_depth_image_object()
        transformed_image_object = transformation.depth_image_to_color_camera(
            depth_image_object, transformed_image_object)
        depth_image = transformed_image_object.to_numpy()
        cv2.convertScaleAbs(depth_image, alpha=0.08, dst=depth_8bit_image)
        cv2.applyColorMap(
            depth_8bit_image, cv2.COLORMAP_CIVIDIS, dst=depth_colorized_image)

        cv2.addWeighted(
            gray_3channel_image, 0.7, depth_colorized_image, 0.3, 0,
            dst=combined_image)
        cv2.imshow("Transformed depth image", combined_image)

        if cv2.waitKey(1) == ord("q"):
            break
        frc.update()

    cv2.destroyAllWindows()
    stop_event.set()
    t.join()
    del device


if __name__ == "__main__":
    main()
