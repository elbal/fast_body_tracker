import cv2
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
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

    device = pykinect.start_device(config=device_config)
    body_tracker = pykinect.start_body_tracker(calibration=device.calibration)

    q = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    t = threading.Thread(
        target=tracking_thread, args=(device, body_tracker, q, stop_event))

    cv2.namedWindow("Color image with skeleton", cv2.WINDOW_NORMAL)
    frc = pykinect.FrameRateCalculator()

    t.start()
    frc.start()
    while True:
        capture, frame = q.get()

        color_image_object = capture.get_color_image_object()
        color_image = color_image_object.to_numpy()

        bodies = frame.get_bodies()
        for body in bodies:
            positions_2d = body.get_2d_positions(
                calibration=device.calibration,
                target_camera=pykinect.K4A_CALIBRATION_TYPE_COLOR)
            pykinect.draw_body(color_image, positions_2d, body.id)

        cv2.imshow("Color image with skeleton", color_image)

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
