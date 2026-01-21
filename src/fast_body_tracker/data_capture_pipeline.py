import threading
import queue
import numpy as np
from numpy import typing as npt
import cv2
import av
import pathlib
import h5py
import tkinter as tk
from datetime import datetime

from .initializer import initialize_libraries, start_device, start_body_tracker
from .utils.performace_calculator import (
    DroppedFramesAlert, FrameRateCalculator)
from .k4a.k4a_const import (
    K4A_CALIBRATION_TYPE_COLOR, K4A_WIRED_SYNC_MODE_STANDALONE,
    K4A_WIRED_SYNC_MODE_MASTER, K4A_WIRED_SYNC_MODE_SUBORDINATE,
    K4A_IMAGE_FORMAT_COLOR_BGRA32, K4A_COLOR_RESOLUTION_1080P,
    K4A_DEPTH_MODE_WFOV_2X2BINNED)
from .k4a.calibration import Calibration
from .k4a.configuration import Configuration
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
        device_id: int, calibration: Calibration,
        capture_queue: queue.Queue, joints_queue: queue.Queue,
        video_queue: queue.Queue, visualization_queue: queue.Queue,
        ext_rot: npt.NDArray[np.float64] | None = None,
        ext_trans: npt.NDArray[np.float64] | None = None):
    dfa = DroppedFramesAlert()

    frame_idx = 0
    while True:
        item = capture_queue.get()
        if item is None:
            break
        capture, frame = item

        color_image_object = capture.get_color_image_object()
        ts = color_image_object.timestamp
        system_ts = color_image_object.system_timestamp
        bgra_image = color_image_object.to_numpy()

        bodies = []
        for body_idx in range(frame.get_num_bodies()):
            body = frame.get_body(body_idx)
            positions_2d = body.get_2d_positions(
                calibration=calibration,
                target_camera=K4A_CALIBRATION_TYPE_COLOR)
            draw_body(bgra_image, positions_2d, body.id)
            if ext_rot is not None:
                body.positions[:] = body.positions @ ext_rot.T
                body.positions[:] += (ext_trans * 1000.0)
            bodies.append(body)

        if joints_queue.full():
            dfa.update()
            try:
                joints_queue.get_nowait()
            except queue.Empty:
                pass
        joints_queue.put((frame_idx, ts, system_ts, device_id, bodies))

        bgr_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR)
        if video_queue.full():
            dfa.update()
            try:
                video_queue.get_nowait()
            except queue.Empty:
                pass
        video_queue.put((bgr_image, device_id))

        if visualization_queue.full():
            dfa.update()
            try:
                visualization_queue.get_nowait()
            except queue.Empty:
                pass
        visualization_queue.put((bgr_image, device_id))

        frame_idx += 1
    joints_queue.put(None)
    video_queue.put(None)
    visualization_queue.put(None)


def body_saver_thread(
        joints_queue: queue.Queue, file_dir: pathlib.Path,
        n_devices: int = 1, n_bodies: int = 1, flush_size: int = 30*60):
    n_joints = len(K4ABT_JOINT_NAMES)
    h5file = h5py.File(file_dir / "body.h5", "w", libver="latest")

    joint_buffers = {
        (i, j): {
            "frame_idx": np.empty(flush_size, dtype=np.int64),
            "positions": np.empty((flush_size, n_joints, 3), dtype=np.float32),
            "confidences": np.empty((flush_size, n_joints), dtype=np.uint8),
            "idx": 0}
        for i in range(n_devices) for j in range(n_bodies)
        }
    ts_buffers = {
        i: {
            "ts": np.empty(flush_size, dtype=np.uint64),
            "system_ts": np.empty(flush_size, dtype=np.uint64),
            "idx": 0}
        for i in range(n_devices)}

    joint_names = np.array(
        K4ABT_JOINT_NAMES, dtype=h5py.string_dtype(encoding="utf-8"))
    h5file.create_dataset("joint_names", data=joint_names)
    h5file.create_dataset(
        "joint_connections", data=K4ABT_SEGMENT_PAIRS, dtype="u1")

    joint_data = {}
    joints_grp = h5file.create_group("joints")
    for i in range(n_devices):
        device_grp = joints_grp.create_group(f"device_{i}")
        device_grp.attrs["device_id"] = i
        for j in range(n_bodies):
            body_grp = device_grp.create_group(f"body_{j}")
            body_grp.attrs["body_idx"] = j
            body_grp.create_dataset(
                "frame_idx", shape=(0,), maxshape=(None,), dtype="i8",
                chunks=(flush_size,))
            body_grp.create_dataset(
                "positions", shape=(0, n_joints, 3), maxshape=(None, n_joints, 3),
                dtype="f4", chunks=(flush_size, n_joints, 3))
            body_grp.create_dataset(
                "confidences", shape=(0, n_joints), maxshape=(None, n_joints),
                dtype="u1", chunks=(flush_size, n_joints))

            joint_data[(i, j)] = {
                "frame_idx": body_grp["frame_idx"],
                "positions": body_grp["positions"],
                "confidences": body_grp["confidences"]}

    ts_data = {}
    ts_grp = h5file.create_group("ts")
    for i in range(n_devices):
        device_grp = ts_grp.create_group(f"device_{i}")
        device_grp.attrs["device_id"] = i
        device_grp.create_dataset(
            "ts", shape=(0,), maxshape=(None,), dtype="u8",
            chunks=(flush_size,))
        device_grp.create_dataset(
            "system_ts", shape=(0,), maxshape=(None,), dtype="u8",
            chunks=(flush_size,))

        ts_data[i] = {
            "ts": device_grp["ts"], "system_ts": device_grp["system_ts"]}

    def flush_joint_buffer(device_id: int, body_idx: int):
        data = joint_data[(device_id, body_idx)]
        buffer = joint_buffers[(device_id, body_idx)]
        idx = buffer["idx"]

        d_frame_idx = data["frame_idx"]
        d_positions = data["positions"]
        d_confidences = data["confidences"]

        old_n = d_frame_idx.shape[0]
        new_n = old_n + idx

        d_frame_idx.resize(new_n, axis=0)
        d_positions.resize(new_n, axis=0)
        d_confidences.resize(new_n, axis=0)

        d_frame_idx[old_n:new_n] = buffer["frame_idx"][:idx]
        d_positions[old_n:new_n, :, :] = buffer["positions"][:idx]
        d_confidences[old_n:new_n, :] = buffer["confidences"][:idx]

        buffer["idx"] = 0

    def flush_ts_buffer(device_id: int):
        data = ts_data[device_id]
        buffer = ts_buffers[device_id]
        idx = buffer["idx"]

        d_ts = data["ts"]
        d_system_ts = data["system_ts"]
        old_n = d_ts.shape[0]
        new_n = old_n + idx
        d_ts.resize(new_n, axis=0)
        d_system_ts.resize(new_n, axis=0)
        d_ts[old_n:new_n] = buffer["ts"][:idx]
        d_system_ts[old_n:new_n] = buffer["system_ts"][:idx]

        buffer["idx"] = 0

    finished_workers = 0
    flush = False
    while finished_workers < n_devices:
        item = joints_queue.get()
        if item is None:
            finished_workers += 1
            continue

        frame_idx, ts, system_ts, device_id, bodies = item
        buffer = ts_buffers[device_id]
        idx = buffer["idx"]

        buffer["ts"][idx] = ts
        buffer["system_ts"][idx] = system_ts
        buffer["idx"] += 1
        if buffer["idx"] >= flush_size:
            flush_ts_buffer(device_id)
            flush = True

        for body_idx, body in enumerate(bodies):
            if body_idx >= n_bodies:
                break
            buffer = joint_buffers[(device_id, body_idx)]
            idx = buffer["idx"]

            buffer["frame_idx"][idx] = frame_idx
            buffer["positions"][idx, :, :] = body.positions
            buffer["confidences"][idx, :] = body.confidences
            buffer["idx"] += 1
            if (buffer["idx"] >= flush_size) or flush:
                flush_joint_buffer(device_id, body_idx)
                flush = True

        if flush:
            h5file.flush()
            flush = False

    flush = False
    for device_id in range(n_devices):
        if ts_buffers[device_id]["idx"] > 0:
            flush_ts_buffer(device_id)
            flush = True
        for body_idx in range(n_bodies):
            if joint_buffers[(device_id, body_idx)]["idx"] > 0:
                flush_joint_buffer(device_id, body_idx)
                flush = True
    if flush:
        h5file.flush()
    h5file.close()


def video_saver_thread(
        video_queue: queue.Queue, video_dir: pathlib.Path, n_devices: int,
        fps: int = 30, width: int = 1920, height: int = 1080):
    containers = {}
    streams = {}
    for i in range(n_devices):
        filename = video_dir / f"device_{i}.mkv"
        container = av.open(str(filename), mode="w")

        stream = container.add_stream("av1_nvenc", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        stream.options = {
            "preset": "p4", "tune": "ll", "rc": "vbr", "cq": "28", "gpu": "0"}
        containers[i] = container
        streams[i] = stream

    finished_workers = 0
    while finished_workers < n_devices:
        item = video_queue.get()
        if item is None:
            finished_workers += 1
            continue

        image, device_id = item
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")

        stream = streams[device_id]
        container = containers[device_id]

        for packet in stream.encode(frame):
            container.mux(packet)

    for i in range(n_devices):
        for packet in streams[i].encode():
            containers[i].mux(packet)
        containers[i].close()


def visualization_main_tread(
        visualization_queue: queue.Queue, stop_event: threading.Event,
        n_devices: int, width: int = 1920, height: int = 1080):
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
        bgr_image, device_id = item

        cv2.imshow(f"Color images with skeleton {device_id}", bgr_image)
        if cv2.waitKey(1) == ord("q"):
            stop_event.set()
    cv2.destroyAllWindows()


def _default_device_initialization(
        device_index: int = 0,
        device_mode: str = "standalone") -> tuple[Device, Tracker]:
    modes = {
        "standalone": K4A_WIRED_SYNC_MODE_STANDALONE,
        "main": K4A_WIRED_SYNC_MODE_MASTER,
        "secondary": K4A_WIRED_SYNC_MODE_SUBORDINATE}

    device_config = Configuration()
    device_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED
    device_config.synchronized_images_only = True
    device_config.wired_sync_mode = modes[device_mode]

    device = start_device(device_index=device_index, config=device_config)
    tracker = start_body_tracker(calibration=device.calibration)

    return device, tracker


def default_pipeline(
        base_dir: pathlib.Path | str,
        trans_matrices: dict[int, npt.NDArray[np.float32]] | None = None,
        sync: bool = False, n_bodies: int = 1):
    if trans_matrices is None:
        n_devices = 1
    else:
        n_devices = len(trans_matrices) + 1

    devices = dict()
    trackers = dict()

    capture_queues = dict()
    capture_t = dict()
    stop_event = threading.Event()

    computation_t = dict()
    joints_queue = queue.Queue(maxsize=10)
    video_queue = queue.Queue(maxsize=10)
    visualization_queue = queue.Queue(maxsize=10)

    initialize_libraries(track_body=True)
    for i in range(n_devices - 1, -1, -1):  # Start the secondary first.
        if sync and n_devices != 1:
            if i == 0:
                device_mode = "main"
            else:
                device_mode = "secondary"
        else:
            device_mode = "standalone"
        device, tracker = _default_device_initialization(
            device_index=i, device_mode=device_mode)
        devices[i] = device
        trackers[i] = tracker

        capture_queues[i] = queue.Queue(maxsize=10)
        capture_t[i] = threading.Thread(
            target=capture_thread,
            args=(device, tracker, capture_queues[i], stop_event))

        if i == 0:
            rot_matrix = None
            trans_vector = None
        else:
            rot_matrix = trans_matrices[i][:3, :3]
            trans_vector = trans_matrices[i][:3, 3]
        computation_t[i] = threading.Thread(
            target=computation_thread,
            args=(
                i, device.calibration, capture_queues[i], joints_queue,
                video_queue, visualization_queue, rot_matrix, trans_vector))

    base_dir = pathlib.Path(base_dir)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    file_dir = base_dir / timestamp
    file_dir.mkdir(parents=True, exist_ok=True)
    body_saver_t = threading.Thread(
        target=body_saver_thread,
        args=(joints_queue, file_dir, n_devices, n_bodies))
    video_saver_t = threading.Thread(
        target=video_saver_thread, args=(video_queue, file_dir, n_devices))

    video_saver_t.start()
    body_saver_t.start()
    for t in computation_t.values():
        t.start()
    for t in capture_t.values():
        t.start()
    visualization_main_tread(visualization_queue, stop_event, n_devices)

    video_saver_t.join()
    body_saver_t.join()
    for t in capture_t.values():
        t.join()
    for t in computation_t.values():
        t.join()
    del trackers
    del devices
