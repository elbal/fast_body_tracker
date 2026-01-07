import cv2
import threading
import queue

import pykinect_azure as pykinect


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
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF

    device = pykinect.start_device(config=device_config)
    q = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    t = threading.Thread(target=capture_thread, args=(device, q, stop_event))

    cv2.namedWindow("Color image", cv2.WINDOW_NORMAL)

    frc = pykinect.FrameRateCalculator()

    t.start()
    frc.start()
    while True:
        capture = q.get()
        image_object = capture.get_color_image_object()
        bgra_image = image_object.to_numpy()
        cv2.imshow("Color image", bgra_image)

        if cv2.waitKey(1) == ord("q"):
            break
        frc.update()
    cv2.destroyAllWindows()
    stop_event.set()
    t.join()
    del device


if __name__ == "__main__":
    main()
