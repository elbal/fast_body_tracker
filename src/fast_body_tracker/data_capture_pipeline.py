import threading
import queue
import time
import cv2
import numpy as np
from numpy import typing as npt

from .utils.performace_calculator import DroppedFramesAlert
from .k4a.k4a_const import K4A_CALIBRATION_TYPE_COLOR
from .k4a.calibration import Calibration
from .k4a.device import Device
from .k4abt.body import draw_body
from .k4abt.tracker import Tracker


def capture_thread(
        device: Device, tracker: Tracker | None, capture_queue: queue.Queue,
        stop_event: threading.Event):
    dfa = DroppedFramesAlert()

    while not stop_event.is_set():
        capture = device.update()
        if tracker is not None:
            frame = tracker.update(capture=capture)
        if capture_queue.full():
            dfa.update()
            try:
                capture_queue.get_nowait()
            except queue.Empty:
                pass
        if tracker is not None:
            capture_queue.put((capture, frame))
        else:
            capture_queue.put(capture)
    capture_queue.put(None)


def computation_thread(
        thread_id: int, calibration: Calibration, capture_queue: queue.Queue,
        image_array: npt.NDArray[np.uint8], image_lock: threading.Lock):
    h = 1080
    s_top = thread_id * h
    s_bot = (thread_id+1) * h

    while True:
        item = capture_queue.get()
        if item is None:
            break
        capture, frame = item

        color_image_object = capture.get_color_image_object()
        color_image = color_image_object.to_numpy()

        if frame.get_num_bodies() > 0:
            body = frame.get_body()
            positions_2d = body.get_2d_positions(
                calibration=calibration,
                target_camera=K4A_CALIBRATION_TYPE_COLOR)
            draw_body(color_image, positions_2d, body.id)

        with image_lock:
            np.copyto(image_array[s_top:s_bot, :, :], color_image)


def visualization_main_tread(
        image_array: npt.NDArray[np.uint8], image_lock: threading.Lock,
        stop_event: threading.Event):
    cv2.namedWindow("Color images with skeleton", cv2.WINDOW_NORMAL)
    with image_lock:
        internal_image = image_array.copy()
    target_interval = 1 / 30

    while not stop_event.is_set():
        start_time = time.perf_counter()

        with image_lock:
            np.copyto(internal_image, image_array)
        cv2.imshow("Color images with skeleton", internal_image)
        if cv2.waitKey(1) == ord("q"):
            stop_event.set()
            break

        elapsed_time = time.perf_counter() - start_time
        sleep_time = target_interval - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
    cv2.destroyAllWindows()
