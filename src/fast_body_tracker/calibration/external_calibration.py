import numpy as np
from numpy import typing as npt
import cv2

from ..initializer import initialize_libraries, start_device
from ..k4a.k4a_const import (
    K4A_COLOR_RESOLUTION_2160P,
    K4A_CALIBRATION_TYPE_COLOR,
    K4A_CALIBRATION_TYPE_DEPTH,
    K4A_DEPTH_MODE_WFOV_2X2BINNED,
)
from ..k4a.configuration import Configuration


_BOARD_CV2_TO_WORLD_ROT = np.diag((1.0, -1.0, -1.0)).astype(np.float32)


def _invert_rigid_transform(
    rot_matrix: npt.NDArray[np.float32], trans_vector: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    inv_rot = rot_matrix.T
    inv_trans = -inv_rot @ trans_vector

    return inv_rot, inv_trans


def external_calibration(
    n_devices, n_samples: int = 60
) -> dict[int, npt.NDArray[np.float32]]:
    n_squares_w = 3
    n_squares_h = 3
    square_length_mm = 94.0
    marker_length_mm = 94.0 * 0.76

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard(
        size=(n_squares_w, n_squares_h),
        squareLength=square_length_mm,
        markerLength=marker_length_mm,
        dictionary=aruco_dict,
    )
    detector = cv2.aruco.CharucoDetector(board)

    initialize_libraries()
    calibration_data = {}
    for idx in range(n_devices):
        configuration = Configuration()
        configuration.color_resolution = K4A_COLOR_RESOLUTION_2160P
        configuration.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED
        device = start_device(device_index=idx, config=configuration)

        k_matrix = device.calibration.get_k_matrix(K4A_CALIBRATION_TYPE_COLOR)
        dist_params = device.calibration.get_dist_params(K4A_CALIBRATION_TYPE_COLOR)

        bgra2depth_rot, bgra2depth_trans = device.calibration.get_extrinsics(
            K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH
        )
        bgra2depth_trans = bgra2depth_trans / 1000.0  # From mm to meters.

        rvecs, tvecs = [], []
        window_name = f"{idx}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while len(rvecs) < n_samples:
            color_image_object = device.update().get_color_image_object()
            if color_image_object is None:
                continue

            bgra_image = color_image_object.to_numpy()
            gray_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2GRAY)
            corners_xy, corners_idx, _, _ = detector.detectBoard(gray_image)

            if corners_idx is not None and len(corners_idx) >= 4:
                main_corners_xyz, main_corners_xy = board.matchImagePoints(
                    corners_xy, corners_idx
                )
                valid, rvec, tvec = cv2.solvePnP(
                    main_corners_xyz, main_corners_xy, k_matrix, dist_params
                )
                if valid:
                    rvecs.append(rvec)
                    tvecs.append(tvec / 1000.0)  # From mm to meters.
                    cv2.aruco.drawDetectedCornersCharuco(
                        bgra_image, corners_xy, corners_idx, (0, 255, 0)
                    )
                    cv2.drawFrameAxes(
                        bgra_image, k_matrix, dist_params, rvec, tvec, 100
                    )

            cv2.putText(
                bgra_image,
                f"Samples: {len(rvecs)}/{n_samples}",
                (50, 80),
                1,
                3,
                (0, 0, 255),
                3,
            )
            cv2.imshow(window_name, bgra_image)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyWindow(window_name)
        if not rvecs:
            raise RuntimeError(
                f"No valid ChArUco board pose detected for device {idx}. "
                "External calibration aborted."
            )

        board2bgra_rot, _ = cv2.Rodrigues(np.median(rvecs, axis=0))
        calibration_data[idx] = {
            "bgra2depth_rot": bgra2depth_rot,
            "bgra2depth_trans": bgra2depth_trans,
            "board2bgra_rot": board2bgra_rot,
            "board2bgra_trans": np.median(tvecs, axis=0).flatten(),
        }

    trans_matrices = dict()

    for idx in range(n_devices):
        device_data = calibration_data[idx]
        depth2bgra_rot, depth2bgra_trans = _invert_rigid_transform(
            device_data["bgra2depth_rot"], device_data["bgra2depth_trans"]
        )
        bgra2board_cv_rot, bgra2board_cv_trans = _invert_rigid_transform(
            device_data["board2bgra_rot"], device_data["board2bgra_trans"]
        )
        depth2board_cv_rot = bgra2board_cv_rot @ depth2bgra_rot
        depth2board_cv_trans = (
            bgra2board_cv_rot @ depth2bgra_trans + bgra2board_cv_trans
        )
        # Rotate 180 degrees around the board X axis so +Z points up while
        # preserving a right-handed frame.
        depth2world_rot = _BOARD_CV2_TO_WORLD_ROT @ depth2board_cv_rot
        depth2world_trans = _BOARD_CV2_TO_WORLD_ROT @ depth2board_cv_trans

        trans_matrix = np.eye(4, dtype=np.float32)
        trans_matrix[:3, :3] = depth2world_rot
        trans_matrix[:3, 3] = depth2world_trans
        trans_matrices[idx] = trans_matrix

    return trans_matrices
