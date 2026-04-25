import cv2
import numpy as np
import threading
import queue

import fast_body_tracker as fbt


def main():
    fbt.initialize_libraries()

    device_config = fbt.Configuration()
    device_config.color_resolution = fbt.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = fbt.K4A_DEPTH_MODE_WFOV_2X2BINNED

    device = fbt.start_device(config=device_config)
    q = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    t = threading.Thread(target=fbt.capture_thread, args=(device, None, q, stop_event))

    cv2.namedWindow("Depth image", cv2.WINDOW_NORMAL)
    depth_8bit_image = np.zeros((512, 512), dtype=np.uint8)
    colorized_image = np.zeros((512, 512, 3), dtype=np.uint8)

    frc = fbt.FrameRateCalculator()

    t.start()
    frc.start()
    while True:
        capture = q.get()
        if capture is None:
            break

        image_object = capture.get_depth_image_object()
        if image_object is None:
            continue
        depth_image = image_object.to_numpy()
        cv2.convertScaleAbs(depth_image, alpha=0.08, dst=depth_8bit_image)
        cv2.applyColorMap(depth_8bit_image, cv2.COLORMAP_CIVIDIS, dst=colorized_image)
        cv2.imshow("Depth image", colorized_image)

        if cv2.waitKey(1) == ord("q"):
            stop_event.set()
        frc.update()
    cv2.destroyAllWindows()
    t.join()
    del device


if __name__ == "__main__":
    main()
