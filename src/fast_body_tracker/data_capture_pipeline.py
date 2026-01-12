import threading
import queue
import numpy as np
import cv2
import av
import pathlib
import h5py
import tkinter as tk

from .utils.performace_calculator import (
    DroppedFramesAlert, FrameRateCalculator)
from .k4a.k4a_const import K4A_CALIBRATION_TYPE_COLOR
from .k4a.calibration import Calibration
from .k4a.device import Device
from .k4abt.kabt_const import K4ABT_JOINT_NAMES, K4ABT_SEGMENT_PAIRS
from .k4abt.body import draw_body
from .k4abt.tracker import Tracker


def capture_thread(
        device: Device, tracker: Tracker | None, capture_queue: queue.Queue,
        stop_event: threading.Event):
    frc = FrameRateCalculator()
    dfa = DroppedFramesAlert()

    frc.start()
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
        frc.update()
    capture_queue.put(None)


def computation_thread(
        device_id: int, calibration: Calibration, capture_queue: queue.Queue,
        joints_queue: queue.Queue, video_queue: queue.Queue,
        visualization_queue: queue.Queue):
    dfa = DroppedFramesAlert()

    frame_idx = 0
    while True:
        item = capture_queue.get()
        if item is None:
            break
        capture, frame = item

        color_image_object = capture.get_color_image_object()
        ts = color_image_object.timestamp
        color_image = color_image_object.to_numpy()

        if frame.get_num_bodies() > 0:
            body = frame.get_body()
            positions_2d = body.get_2d_positions(
                calibration=calibration,
                target_camera=K4A_CALIBRATION_TYPE_COLOR)
            draw_body(color_image, positions_2d, body.id)
            if joints_queue.full():
                dfa.update()
                try:
                    joints_queue.get_nowait()
                except queue.Empty:
                    pass
            joints_queue.put((body, ts, frame_idx, device_id))

        if video_queue.full():
            dfa.update()
            try:
                video_queue.get_nowait()
            except queue.Empty:
                pass
        video_queue.put((color_image, device_id))

        if visualization_queue.full():
            dfa.update()
            try:
                visualization_queue.get_nowait()
            except queue.Empty:
                pass
        visualization_queue.put((color_image, device_id))
    joints_queue.put(None)
    video_queue.put(None)
    visualization_queue.put(None)


def joints_saver_thread(
        n_devices: int, joints_queue: queue.Queue, file_dir: pathlib.Path,
        flush_size: int = 600):
    n_joints = len(K4ABT_JOINT_NAMES)
    h5file = h5py.File(file_dir / "joints.h5", "w")

    buffers = {
        i: {
            "ts": np.empty(flush_size, dtype=np.int64),
            "positions": np.empty((flush_size, n_joints, 3), dtype=np.float32),
            "confidences": np.empty((flush_size, n_joints), dtype=np.uint8),
            "frame_idx": np.empty(flush_size, dtype=np.int64),
            "idx": 0
            } for i in range(n_devices)
        }

    joint_names = np.array(
        K4ABT_JOINT_NAMES, dtype=h5py.string_dtype(encoding="utf-8"))
    h5file.create_dataset("joint_names", data=joint_names)
    h5file.create_dataset(
        "joint_connections", data=K4ABT_SEGMENT_PAIRS, dtype="u1")

    device_groups = {}
    for i in range(n_devices):
        grp = h5file.create_group(f"device_{i}")
        grp.attrs["device_idx"] = i
        grp.create_dataset(
            "ts", shape=(0,), maxshape=(None,), dtype="i8",
            chunks=(flush_size,))
        grp.create_dataset(
            "positions", shape=(0, n_joints, 3), maxshape=(None, n_joints, 3),
            dtype="f4", chunks=(flush_size, n_joints, 3))
        grp.create_dataset(
            "confidences", shape=(0, n_joints), maxshape=(None, n_joints),
            dtype="u1", chunks=(flush_size, n_joints))
        grp.create_dataset(
            "frame_idx", shape=(0,), maxshape=(None,), dtype="i8",
            chunks=(flush_size,))
        device_groups[i] = grp

    def flush_device_buffer(device_id: int):
        grp = device_groups[device_id]
        buffer = buffers[device_id]
        idx = buffer["idx"]

        d_ts = grp["ts"]
        d_positions = grp["positions"]
        d_confidences = grp["confidences"]
        d_frame_idx = grp["frame_idx"]

        old_n = d_ts.shape[0]
        new_n = old_n + idx

        d_ts.resize(new_n, axis=0)
        d_positions.resize(new_n, axis=0)
        d_confidences.resize(new_n, axis=0)
        d_frame_idx.resize(new_n, axis=0)

        d_ts[old_n:new_n] = buffer["ts"][:idx]
        d_positions[old_n:new_n, :, :] = buffer["positions"][:idx]
        d_confidences[old_n:new_n, :] = buffer["confidences"][:idx]
        d_frame_idx[old_n:new_n] = buffer["frame_idx"][:idx]

        h5file.flush()
        buffer["idx"] = 0

    finished_workers = 0
    while finished_workers < n_devices:
        item = joints_queue.get()
        if item is None:
            finished_workers += 1
            continue

        body, ts, frame_idx, device_id = item
        buffer = buffers[device_id]
        idx = buffer["idx"]

        buffer["ts"][idx] = ts
        buffer["positions"][idx, :, :] = body.positions
        buffer["confidences"][idx, :] = body.confidences
        buffer["frame_idx"][idx] = frame_idx
        buffer["idx"] += 1

        if buffer["idx"] >= flush_size:
            flush_device_buffer(device_id)
    for i in range(n_devices):
        if buffers[i]["idx"] > 0:
            flush_device_buffer(i)
    h5file.close()


def video_saver_thread(
        n_devices: int, video_queue: queue.Queue, video_dir: pathlib.Path,
        fps: int = 30, width: int = 1920, height: int = 1080):
    containers = {}
    streams = {}
    for i in range(n_devices):
        filename = video_dir / f"device_{i}.mkv"
        container = av.open(str(filename), mode="w")
        stream = container.add_stream("hevc_nvenc", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "preset": "p4", "tune": "ll", "rc": "vbr", "cq": "24", "gpu": "0"}
        containers[i] = container
        streams[i] = stream

    finished_workers = 0
    while finished_workers < n_devices:
        item = video_queue.get()
        if item is None:
            finished_workers += 1
            continue

        bgra_image, device_id = item
        bgr_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR)
        frame = av.VideoFrame.from_ndarray(bgr_image, format="bgr24")

        stream = streams[device_id]
        container = containers[device_id]

        for packet in stream.encode(frame):
            container.mux(packet)

    for i in range(n_devices):
        for packet in streams[i].encode():
            containers[i].mux(packet)
        containers[i].close()


def visualization_main_tread(
        n_devices: int, visualization_queue: queue.Queue,
        stop_event: threading.Event, width: int = 1920, height: int = 1080):
    window_bar_height = 20
    taskbar_height = 30
    from_border = 5
    aspect_ratio = width / height

    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    window_h = (screen_h-taskbar_height)//n_devices - window_bar_height
    window_w = int(window_h * aspect_ratio)

    for i in range(n_devices):
        window_name = f"Color images with skeleton {i}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_w, window_h)
        cv2.moveWindow(
            window_name, screen_w - window_w - from_border, i*window_h)

    while True:
        item = visualization_queue.get()
        if item is None:
            break
        image, device_id = item

        cv2.imshow(f"Color images with skeleton {device_id}", image)
        if cv2.waitKey(1) == ord("q"):
            stop_event.set()
    cv2.destroyAllWindows()
