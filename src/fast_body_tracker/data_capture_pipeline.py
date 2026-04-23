import threading
import queue
from dataclasses import dataclass
import numpy as np
from fast_body_tracker import Body
from numpy import typing as npt
import cv2
import av
import pathlib
import h5py
import tkinter as tk
from datetime import datetime

from .k4abt.kabt_const import K4ABT_JOINT_PELVIS

from .initializer import initialize_libraries, start_device, start_body_tracker
from .utils.performace_calculator import DroppedFramesAlert, FrameRateCalculator
from .k4a.k4a_const import (
    K4A_CALIBRATION_TYPE_COLOR,
    K4A_WIRED_SYNC_MODE_STANDALONE,
    K4A_WIRED_SYNC_MODE_MASTER,
    K4A_WIRED_SYNC_MODE_SUBORDINATE,
    K4A_IMAGE_FORMAT_COLOR_BGRA32,
    K4A_COLOR_RESOLUTION_1080P,
    K4A_DEPTH_MODE_WFOV_2X2BINNED,
)
from .k4a.calibration import Calibration
from .k4a.configuration import Configuration
from .k4a.device import Device
from .k4abt.kabt_const import K4ABT_JOINT_NAMES, K4ABT_SEGMENT_PAIRS
from .k4abt.body import draw_body
from .k4abt.tracker import Tracker


def capture_thread(
    device: Device,
    tracker: Tracker | None,
    capture_queue: queue.Queue,
    stop_event: threading.Event,
):
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
    device_id: int,
    calibration: Calibration,
    capture_queue: queue.Queue,
    unification_queue: queue.Queue,
    video_queue: queue.Queue,
    visualization_queue: queue.Queue,
    ext_rot: npt.NDArray[np.float64] | None = None,
    ext_trans: npt.NDArray[np.float64] | None = None,
):
    dfa = DroppedFramesAlert()

    frame_idx = 0
    while True:
        item = capture_queue.get()
        if item is None:
            break
        capture, frame = item
        color_image_object = capture.get_color_image_object()
        bgra_image = color_image_object.to_numpy()

        bodies = []
        n_detected_bodies = frame.get_num_bodies()
        for body_idx in range(n_detected_bodies):
            body = frame.get_body(body_idx)
            positions_2d = body.get_2d_positions(
                calibration=calibration, target_camera=K4A_CALIBRATION_TYPE_COLOR
            )
            draw_body(bgra_image, positions_2d, body.id)
            if ext_rot is not None:
                body.positions[:] = body.positions @ ext_rot.T
                body.positions[:] += ext_trans * 1000.0
            bodies.append(body)

        if device_id == 0 or n_detected_bodies > 0:
            ts = color_image_object.timestamp
            system_ts = color_image_object.system_timestamp
            if unification_queue.full():
                dfa.update()
                try:
                    unification_queue.get_nowait()
                except queue.Empty:
                    pass
            unification_queue.put((device_id, frame_idx, ts, system_ts, bodies))

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
    unification_queue.put(None)
    video_queue.put(None)
    visualization_queue.put((None, device_id))


def _assign_nearest(
    tracked_joints: npt.NDArray[np.float32],
    joints_to_be_assigned: npt.NDArray[np.float32],
    max_distance: float,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    valid_idx = np.flatnonzero(np.isfinite(tracked_joints[:, 0]))
    tracked_valid = tracked_joints[valid_idx]
    diff = tracked_valid[:, np.newaxis, :] - joints_to_be_assigned[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    max_distance_sq = max_distance * max_distance
    candidate_rows, candidate_cols = np.nonzero(dist_sq <= max_distance_sq)
    if candidate_rows.size == 0:
        empty = np.empty(0, dtype=np.int64)
        n_to_assign = joints_to_be_assigned.shape[0]
        return empty, empty, np.arange(n_to_assign, dtype=np.int64)

    assigned_full = np.full(tracked_joints.shape[0], -1, dtype=np.int64)
    row_used = np.zeros(tracked_valid.shape[0], dtype=bool)
    col_used = np.zeros(joints_to_be_assigned.shape[0], dtype=bool)
    max_matches = min(tracked_valid.shape[0], joints_to_be_assigned.shape[0])
    assigned_count = 0
    order = np.argsort(dist_sq[candidate_rows, candidate_cols], kind="quicksort")
    for k in order:
        row_idx = candidate_rows[k]
        col_idx = candidate_cols[k]
        if row_used[row_idx] or col_used[col_idx]:
            continue

        row_used[row_idx] = True
        col_used[col_idx] = True
        assigned_full[valid_idx[row_idx]] = col_idx
        assigned_count += 1
        if assigned_count == max_matches:
            break

    assigned_to_idx = np.flatnonzero(assigned_full >= 0)
    assigned_idx = assigned_full[assigned_to_idx]
    unassigned_idx = np.flatnonzero(~col_used)

    return assigned_idx, assigned_to_idx, unassigned_idx


@dataclass(slots=True)
class _TrackingPool:
    tracked_joints: npt.NDArray[np.float32]
    tags: npt.NDArray[np.int64]
    available_slots: list[int]
    stale_counter: npt.NDArray[np.uint8]
    next_tag: int


@dataclass(slots=True)
class _CurrentFrame:
    idx: int | None
    ts: int | None
    system_ts: int | None
    bodies: list[Body | None]
    tags: npt.NDArray[np.int64]
    contributions: npt.NDArray[np.uint64]


@dataclass(slots=True)
class _Stored:
    bodies: list[list[Body] | None]
    ts: npt.NDArray[np.uint64]


def _update_tracked(
    bodies: list[Body],
    device_id: int,
    tracking_pool: _TrackingPool,
    current_frame: _CurrentFrame,
    is_stale: npt.NDArray[np.bool],
    reference: int,
    max_distance: float,
):
    tracked_joints = tracking_pool.tracked_joints
    tags = tracking_pool.tags
    available_slots = tracking_pool.available_slots
    current_bodies = current_frame.bodies
    current_tags = current_frame.tags
    contributions = current_frame.contributions
    next_tag = tracking_pool.next_tag
    n_bodies = tracked_joints.shape[0]
    n_available_slots = len(available_slots)
    if n_available_slots < n_bodies:
        joints_to_be_assigned = np.empty((len(bodies), 3), dtype=np.float32)
        for i, body in enumerate(bodies):
            joints_to_be_assigned[i] = body.positions[reference]
        assigned_idx, assigned_to_idx, unassigned_idx = _assign_nearest(
            tracked_joints, joints_to_be_assigned, max_distance
        )

        for i, j in zip(assigned_to_idx, assigned_idx):
            body = bodies[j]
            current_body = current_bodies[i]
            if current_body is None:
                tracked_joints[i] = body.positions[reference]
                current_bodies[i] = body
                current_tags[i] = tags[i]
                contributions[i, device_id] += 1
                is_stale[i] = False
            else:
                positions = body.positions
                orientations = body.orientations
                confidences = body.confidences

                confidence_mask = confidences > current_body.confidences
                current_body.positions[confidence_mask] = positions[confidence_mask]
                current_body.orientations[confidence_mask] = orientations[
                    confidence_mask
                ]
                current_body.confidences[confidence_mask] = confidences[confidence_mask]

                tracked_joints[i] = current_body.positions[reference]
                current_tags[i] = tags[i]
                if np.any(confidence_mask):
                    contributions[i, device_id] += 1
    else:
        unassigned_idx = range(len(bodies))

    n_to_fill = min(n_available_slots, len(unassigned_idx))
    for j in unassigned_idx[:n_to_fill]:
        i = available_slots.pop()
        tags[i] = next_tag
        next_tag += 1
        body = bodies[j]
        current_body = current_bodies[i]
        if current_body is None:
            tracked_joints[i] = body.positions[reference]
            current_bodies[i] = body
            current_tags[i] = tags[i]
            contributions[i, device_id] += 1
            is_stale[i] = False
        else:
            positions = body.positions
            orientations = body.orientations
            confidences = body.confidences

            confidence_mask = confidences > current_body.confidences
            current_body.positions[confidence_mask] = positions[confidence_mask]
            current_body.orientations[confidence_mask] = orientations[confidence_mask]
            current_body.confidences[confidence_mask] = confidences[confidence_mask]

            tracked_joints[i] = current_body.positions[reference]
            current_tags[i] = tags[i]
            if np.any(confidence_mask):
                contributions[i, device_id] += 1

    tracking_pool.next_tag = next_tag


def unification_thread(
    unification_queue: queue.Queue,
    save_queue: queue.Queue,
    n_devices: int = 1,
    n_bodies: int = 1,
):
    max_ts_diff = 1 / 30 * 0.5 * 1e6  # 0.5 frames at 30 FPS, in microseconds.
    max_distance = 300.0  # In mmm.
    max_stale_frames = 60
    reference_joint = K4ABT_JOINT_PELVIS

    tracking_pool = _TrackingPool(
        tracked_joints=np.full((n_bodies, 3), np.nan, dtype=np.float32),
        tags=np.full(n_bodies, -1, dtype=np.int64),
        available_slots=list(range(n_bodies)),
        stale_counter=np.full(n_bodies, 0, dtype=np.uint8),
        next_tag=0,
    )
    is_stale = np.full(n_bodies, True, dtype=bool)
    current_frame = _CurrentFrame(
        idx=None,
        ts=None,
        system_ts=None,
        bodies=[None] * n_bodies,
        tags=np.full(n_bodies, -1, dtype=np.int64),
        contributions=np.zeros((n_bodies, n_devices), dtype=np.uint64),
    )
    stored = _Stored(
        bodies=[None] * n_devices,
        ts=np.full(n_devices, 0, dtype=np.uint64),
    )

    finished_workers = 0
    while finished_workers < n_devices:
        item = unification_queue.get()
        if item is None:
            finished_workers += 1
            continue
        device_id, frame_idx, ts, system_ts, bodies = item

        if device_id == 0:
            if current_frame.ts is not None:
                save_queue.put(current_frame)

                tracking_pool.stale_counter[~is_stale] = 0
                tracking_pool.stale_counter[is_stale] += 1
                drop_mask = tracking_pool.stale_counter > max_stale_frames
                tracking_pool.available_slots.extend(np.flatnonzero(drop_mask).tolist())
                tracking_pool.stale_counter[drop_mask] = 0
                tracking_pool.tracked_joints[drop_mask] = np.nan
                tracking_pool.tags[drop_mask] = -1

            current_frame = _CurrentFrame(
                idx=frame_idx,
                ts=ts,
                system_ts=system_ts,
                bodies=[None] * n_bodies,
                tags=np.full(n_bodies, -1, dtype=np.int64),
                contributions=np.zeros((n_bodies, n_devices), dtype=np.uint64),
            )
            is_stale = np.isfinite(tracking_pool.tracked_joints[:, 0])
            if bodies:
                _update_tracked(
                    bodies,
                    device_id,
                    tracking_pool,
                    current_frame,
                    is_stale,
                    reference_joint,
                    max_distance,
                )

            for stored_device_id, (bodies, ts) in enumerate(
                zip(stored.bodies, stored.ts)
            ):
                if bodies is None or np.abs(current_frame.ts - ts) > max_ts_diff:
                    continue
                _update_tracked(
                    bodies,
                    stored_device_id,
                    tracking_pool,
                    current_frame,
                    is_stale,
                    reference_joint,
                    max_distance,
                )
            stored.bodies[:] = [None] * n_devices
            continue

        if current_frame.ts is None:
            stored.bodies[device_id] = bodies
            stored.ts[device_id] = ts
            continue
        if current_frame.ts - ts > max_ts_diff:
            continue
        if ts - current_frame.ts > max_ts_diff:
            stored.bodies[device_id] = bodies
            stored.ts[device_id] = ts
            continue
        _update_tracked(
            bodies,
            device_id,
            tracking_pool,
            current_frame,
            is_stale,
            reference_joint,
            max_distance,
        )

    save_queue.put(None)


def saver_thread(
    save_queue: queue.Queue,
    file_dir: pathlib.Path,
    n_devices: int = 1,
    n_bodies: int = 1,
    flush_size: int = 30 * 60,
):
    n_joints = len(K4ABT_JOINT_NAMES)
    h5file = h5py.File(file_dir / "body.h5", "w", libver="latest")

    joint_buffers = {
        j: {
            "frame_idx": np.empty(flush_size, dtype=np.int64),
            "tags": np.empty(flush_size, dtype=np.int64),
            "positions": np.empty((flush_size, n_joints, 3), dtype=np.float32),
            "confidences": np.empty((flush_size, n_joints), dtype=np.uint8),
            "contributions": np.empty((flush_size, n_devices), dtype=np.uint64),
            "idx": 0,
        }
        for j in range(n_bodies)
    }
    ts_buffer = {
        "frame_idx": np.empty(flush_size, dtype=np.int64),
        "ts": np.empty(flush_size, dtype=np.uint64),
        "system_ts": np.empty(flush_size, dtype=np.uint64),
        "idx": 0,
    }

    joint_names = np.array(K4ABT_JOINT_NAMES, dtype=h5py.string_dtype(encoding="utf-8"))
    h5file.create_dataset("joint_names", data=joint_names)
    h5file.create_dataset("joint_connections", data=K4ABT_SEGMENT_PAIRS, dtype="u1")

    joint_data = {}
    joints_grp = h5file.create_group("joints")
    for j in range(n_bodies):
        body_grp = joints_grp.create_group(f"body_{j}")
        body_grp.attrs["body_idx"] = j
        body_grp.create_dataset(
            "frame_idx",
            shape=(0,),
            maxshape=(None,),
            dtype="i8",
            chunks=(flush_size,),
        )
        body_grp.create_dataset(
            "tags",
            shape=(0,),
            maxshape=(None,),
            dtype="i8",
            chunks=(flush_size,),
        )
        body_grp.create_dataset(
            "positions",
            shape=(0, n_joints, 3),
            maxshape=(None, n_joints, 3),
            dtype="f4",
            chunks=(flush_size, n_joints, 3),
        )
        body_grp.create_dataset(
            "confidences",
            shape=(0, n_joints),
            maxshape=(None, n_joints),
            dtype="u1",
            chunks=(flush_size, n_joints),
        )
        body_grp.create_dataset(
            "contributions",
            shape=(0, n_devices),
            maxshape=(None, n_devices),
            dtype="u8",
            chunks=(flush_size, n_devices),
        )

        joint_data[j] = {
            "frame_idx": body_grp["frame_idx"],
            "tags": body_grp["tags"],
            "positions": body_grp["positions"],
            "confidences": body_grp["confidences"],
            "contributions": body_grp["contributions"],
        }

    ts_grp = h5file.create_group("ts")
    ts_grp.create_dataset(
        "frame_idx", shape=(0,), maxshape=(None,), dtype="i8", chunks=(flush_size,)
    )
    ts_grp.create_dataset(
        "ts", shape=(0,), maxshape=(None,), dtype="u8", chunks=(flush_size,)
    )
    ts_grp.create_dataset(
        "system_ts", shape=(0,), maxshape=(None,), dtype="u8", chunks=(flush_size,)
    )
    ts_data = {
        "frame_idx": ts_grp["frame_idx"],
        "ts": ts_grp["ts"],
        "system_ts": ts_grp["system_ts"],
    }

    def flush_joint_buffer(body_idx: int):
        data = joint_data[body_idx]
        buffer = joint_buffers[body_idx]
        idx = buffer["idx"]

        d_frame_idx = data["frame_idx"]
        d_tags = data["tags"]
        d_positions = data["positions"]
        d_confidences = data["confidences"]
        d_contributions = data["contributions"]

        old_n = d_frame_idx.shape[0]
        new_n = old_n + idx

        d_frame_idx.resize(new_n, axis=0)
        d_tags.resize(new_n, axis=0)
        d_positions.resize(new_n, axis=0)
        d_confidences.resize(new_n, axis=0)
        d_contributions.resize(new_n, axis=0)

        d_frame_idx[old_n:new_n] = buffer["frame_idx"][:idx]
        d_tags[old_n:new_n] = buffer["tags"][:idx]
        d_positions[old_n:new_n, :, :] = buffer["positions"][:idx]
        d_confidences[old_n:new_n, :] = buffer["confidences"][:idx]
        d_contributions[old_n:new_n, :] = buffer["contributions"][:idx]

        buffer["idx"] = 0

    def flush_ts_buffer():
        data = ts_data
        buffer = ts_buffer
        idx = buffer["idx"]

        d_frame_idx = data["frame_idx"]
        d_ts = data["ts"]
        d_system_ts = data["system_ts"]
        old_n = d_frame_idx.shape[0]
        new_n = old_n + idx
        d_frame_idx.resize(new_n, axis=0)
        d_ts.resize(new_n, axis=0)
        d_system_ts.resize(new_n, axis=0)
        d_frame_idx[old_n:new_n] = buffer["frame_idx"][:idx]
        d_ts[old_n:new_n] = buffer["ts"][:idx]
        d_system_ts[old_n:new_n] = buffer["system_ts"][:idx]

        buffer["idx"] = 0

    flush = False
    while True:
        item = save_queue.get()
        if item is None:
            break

        frame = item
        buffer = ts_buffer
        idx = buffer["idx"]

        buffer["frame_idx"][idx] = frame.idx
        buffer["ts"][idx] = frame.ts
        buffer["system_ts"][idx] = frame.system_ts
        buffer["idx"] += 1
        if buffer["idx"] >= flush_size:
            flush_ts_buffer()
            flush = True

        for body_idx, (body, tag) in enumerate(zip(frame.bodies, frame.tags)):
            if body is None or tag < 0:
                continue
            buffer = joint_buffers[body_idx]
            idx = buffer["idx"]

            buffer["frame_idx"][idx] = frame.idx
            buffer["tags"][idx] = tag
            buffer["positions"][idx, :, :] = body.positions
            buffer["confidences"][idx, :] = body.confidences
            buffer["contributions"][idx, :] = frame.contributions[body_idx]
            buffer["idx"] += 1
            if (buffer["idx"] >= flush_size) or flush:
                flush_joint_buffer(body_idx)
                flush = True

        if flush:
            h5file.flush()
            flush = False

    flush = False
    if ts_buffer["idx"] > 0:
        flush_ts_buffer()
        flush = True
    for body_idx in range(n_bodies):
        if joint_buffers[body_idx]["idx"] > 0:
            flush_joint_buffer(body_idx)
            flush = True
    if flush:
        h5file.flush()
    h5file.close()


def video_saver_thread(
    video_queue: queue.Queue,
    video_dir: pathlib.Path,
    n_devices: int,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
):
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
            "preset": "p4",
            "tune": "ll",
            "rc": "vbr",
            "cq": "28",
            "gpu": "0",
        }
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
    visualization_queue: queue.Queue,
    stop_event: threading.Event,
    n_devices: int,
    width: int = 1920,
    height: int = 1080,
):
    window_bar_height = 20
    taskbar_height = 30
    from_border = 5
    aspect_ratio = width / height

    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    window_h = (screen_h - taskbar_height) // n_devices - window_bar_height
    window_w = int(window_h * aspect_ratio)

    for i in range(n_devices):
        window_name = f"Color images with skeleton {i}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_w, window_h)
        cv2.moveWindow(window_name, screen_w - window_w - from_border, i * window_h)

    finished_workers = 0
    while finished_workers < n_devices:
        bgr_image, device_id = visualization_queue.get()
        if bgr_image is None:
            cv2.destroyWindow(f"Color images with skeleton {device_id}")
            finished_workers += 1
            continue

        cv2.imshow(f"Color images with skeleton {device_id}", bgr_image)
        if cv2.waitKey(1) == ord("q"):
            stop_event.set()
    cv2.destroyAllWindows()


def _default_device_initialization(
    device_index: int = 0, device_mode: str = "standalone"
) -> tuple[Device, Tracker]:
    modes = {
        "standalone": K4A_WIRED_SYNC_MODE_STANDALONE,
        "main": K4A_WIRED_SYNC_MODE_MASTER,
        "secondary": K4A_WIRED_SYNC_MODE_SUBORDINATE,
    }

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
    sync: bool = False,
    n_bodies: int = 1,
):
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
    unification_queue = queue.Queue(maxsize=10)
    save_queue = queue.Queue(maxsize=10)
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
            device_index=i, device_mode=device_mode
        )
        devices[i] = device
        trackers[i] = tracker

        capture_queues[i] = queue.Queue(maxsize=10)
        capture_t[i] = threading.Thread(
            target=capture_thread, args=(device, tracker, capture_queues[i], stop_event)
        )

        if i == 0:
            rot_matrix = None
            trans_vector = None
        else:
            rot_matrix = trans_matrices[i][:3, :3]
            trans_vector = trans_matrices[i][:3, 3]
        computation_t[i] = threading.Thread(
            target=computation_thread,
            args=(
                i,
                device.calibration,
                capture_queues[i],
                unification_queue,
                video_queue,
                visualization_queue,
                rot_matrix,
                trans_vector,
            ),
        )

    base_dir = pathlib.Path(base_dir)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    file_dir = base_dir / timestamp
    file_dir.mkdir(parents=True, exist_ok=True)
    unification_t = threading.Thread(
        target=unification_thread,
        args=(unification_queue, save_queue, n_devices, n_bodies),
    )
    saver_t = threading.Thread(
        target=saver_thread, args=(save_queue, file_dir, n_devices, n_bodies)
    )
    video_saver_t = threading.Thread(
        target=video_saver_thread, args=(video_queue, file_dir, n_devices)
    )

    video_saver_t.start()
    unification_t.start()
    saver_t.start()
    for t in computation_t.values():
        t.start()
    for t in capture_t.values():
        t.start()
    visualization_main_tread(visualization_queue, stop_event, n_devices)

    video_saver_t.join()
    unification_t.join()
    saver_t.join()
    for t in capture_t.values():
        t.join()
    for t in computation_t.values():
        t.join()
    del trackers
    del devices
