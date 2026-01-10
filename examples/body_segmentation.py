import cv2
import numpy as np
import queue
import threading

import fast_body_tracker as pykinect


def tracking_thread(device, body_tracker, q, stop_event):
    dfa = pykinect.DroppedFramesAlert()
    while not stop_event.is_set():
        capture = device.update()
        frame = body_tracker.update(capture=capture)
        if q.full():
            dfa.update()
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put((capture, frame))


def main():
    pykinect.initialize_libraries(track_body=True)

    device_config = pykinect.Configuration()
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

    device = pykinect.start_device(config=device_config)
    body_tracker = pykinect.start_body_tracker(calibration=device.calibration)

    q = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    t = threading.Thread(
        target=tracking_thread, args=(device, body_tracker, q, stop_event))

    cv2.namedWindow("Segmented depth image", cv2.WINDOW_NORMAL)
    frc = pykinect.FrameRateCalculator()

    depth_8bit_image = np.zeros((512, 512), dtype=np.uint8)
    depth_colorized_image = np.zeros((512, 512, 3), dtype=np.uint8)
    combined_image = np.zeros((512, 512, 3), dtype=np.uint8)

    t.start()
    frc.start()
    while True:
        capture, frame = q.get()

        image_object = capture.get_depth_image_object()
        depth_image = image_object.to_numpy()
        cv2.convertScaleAbs(depth_image, alpha=0.08, dst=depth_8bit_image)
        cv2.applyColorMap(
            depth_8bit_image, cv2.COLORMAP_CIVIDIS, dst=depth_colorized_image)

        seg_image_object = frame.get_segmentation_image_object()
        rgb_seg_image = pykinect.colorize_segmentation_image(seg_image_object)

        combined_image = cv2.addWeighted(
            rgb_seg_image, 0.6, depth_colorized_image, 0.4, 0,
            dst=combined_image)
        cv2.imshow("Segmented depth image", combined_image)

        if cv2.waitKey(1) == ord("q"):
            break
        frc.update()
    cv2.destroyAllWindows()
    stop_event.set()
    t.join()
    del body_tracker
    del device


if __name__ == "__main__":
    main()
