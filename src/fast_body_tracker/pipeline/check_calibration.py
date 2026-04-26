import time

import matplotlib
import numpy as np
from numpy import typing as npt

from ..initializer import initialize_libraries, start_body_tracker, start_device
from ..k4a.configuration import Configuration
from ..k4a.device import Device
from ..k4a.k4a_const import (
    K4A_COLOR_RESOLUTION_OFF,
    K4A_DEPTH_MODE_WFOV_2X2BINNED,
)

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt


def check_calibration(
    trans_matrices: dict[int, npt.NDArray[np.float32]],
    delay_s: float = 5.0,
) -> None:
    initialize_libraries(track_body=True)
    n_devices = Device.device_get_installed_count()
    if n_devices == 0:
        raise RuntimeError("No devices detected.")

    expected_ids = set(range(n_devices))
    actual_ids = set(trans_matrices)
    if actual_ids != expected_ids:
        problems = []
        missing_ids = sorted(expected_ids - actual_ids)
        unexpected_ids = sorted(actual_ids - expected_ids)
        if missing_ids:
            problems.append(f"missing device ids: {missing_ids}")
        if unexpected_ids:
            problems.append(f"unexpected device ids: {unexpected_ids}")
        raise ValueError(
            "Invalid external calibration matrices: " + ", ".join(problems)
        )
    for device_id, trans_matrix in trans_matrices.items():
        if trans_matrix.shape != (4, 4):
            raise ValueError(
                f"Invalid external calibration matrix for device {device_id}: "
                f"expected shape (4, 4), got {trans_matrix.shape}."
            )

    print(f"Get in position. Capturing in {delay_s:.1f} s.")
    time.sleep(delay_s)

    positions = np.empty((n_devices, 32, 3), dtype=np.float32)
    for device_id in range(n_devices):
        device_config = Configuration()
        device_config.color_resolution = K4A_COLOR_RESOLUTION_OFF
        device_config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.synchronized_images_only = False

        device = start_device(device_index=device_id, config=device_config)
        tracker = start_body_tracker(calibration=device.calibration)

        try:
            capture = device.update()
            frame = tracker.update(capture=capture)
            if frame.get_num_bodies() == 0:
                raise RuntimeError(f"No body detected for device {device_id}.")

            body = frame.get_body(0)
            rot_matrix = trans_matrices[device_id][:3, :3]
            trans_vector = trans_matrices[device_id][:3, 3]
            positions[device_id] = body.positions.copy() @ rot_matrix.T
            positions[device_id] += trans_vector * 1000.0
        finally:
            del tracker
            del device

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")

    for device_id in range(n_devices):
        color_idx = device_id
        color = cmap(color_idx % 10)
        ax.scatter(
            positions[device_id, :, 0],
            positions[device_id, :, 1],
            positions[device_id, :, 2],
            color=[color],
            s=35,
            label=f"device {device_id}",
        )

    points = positions.reshape(-1, 3)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) * 0.5
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0.0:
        radius = 0.5
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.legend()

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("Calibration check")
    plt.tight_layout()
    plt.show()
