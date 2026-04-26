import cv2

from ..initializer import initialize_libraries, start_device
from ..k4a.configuration import Configuration
from ..k4a.device import Device
from ..k4a.k4a_const import (
    K4A_COLOR_RESOLUTION_1080P,
    K4A_DEPTH_MODE_OFF,
    K4A_IMAGE_FORMAT_COLOR_BGRA32,
    K4A_WIRED_SYNC_MODE_STANDALONE,
)


def _draw_device_label(bgra_image, device_id: int):
    image_h = bgra_image.shape[0]
    text = f"device {device_id}"
    org = (20, image_h - 20)

    cv2.putText(
        bgra_image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgra_image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def show_devices():
    initialize_libraries()
    n_devices = Device.device_get_installed_count()
    if n_devices == 0:
        raise RuntimeError("No devices detected.")

    device_config = Configuration()
    device_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = K4A_DEPTH_MODE_OFF
    device_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE

    for device_id in range(n_devices):
        device = start_device(device_index=device_id, config=device_config)
        window_name = f"device {device_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while True:
                color_image_object = device.update().get_color_image_object()
                if color_image_object is None:
                    continue

                bgra_image = color_image_object.to_numpy()
                _draw_device_label(bgra_image, device_id)
                cv2.imshow(window_name, bgra_image)
                if cv2.waitKey(1) == ord("q"):
                    break
        finally:
            cv2.destroyWindow(window_name)
            del device
    cv2.destroyAllWindows()
