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

        board2dev_rot, _ = cv2.Rodrigues(np.median(rvecs, axis=0))
        calibration_data[idx] = {
            "bgra2depth_rot": bgra2depth_rot,
            "bgra2depth_trans": bgra2depth_trans,
            "board2dev_rot": board2dev_rot,
            "board2dev_trans": np.median(tvecs, axis=0).flatten(),
        }

    reference_data = calibration_data[0]
    trans_matrices = dict()

    for i in [idx for idx in range(n_devices) if idx != 0]:
        device_data = calibration_data[i]
        # Secondary RGBA -> main RGBA.
        sec2main_rgba_rot = (
            reference_data["board2dev_rot"] @ device_data["board2dev_rot"].T
        )
        sec2main_rgba_trans = reference_data["board2dev_trans"] - (
            sec2main_rgba_rot @ device_data["board2dev_trans"]
        )
        # Secondary depth -> secondary RGBA.
        depth2rgba_sec_rot = device_data["bgra2depth_rot"].T
        depth2rgba_sec_trans = -depth2rgba_sec_rot @ device_data["bgra2depth_trans"]
        # Secondary depth -> main depth.
        sec2main_depth_rot = (
            reference_data["bgra2depth_rot"] @ sec2main_rgba_rot @ depth2rgba_sec_rot
        )
        sec2main_depth_trans = (
            reference_data["bgra2depth_rot"]
            @ (sec2main_rgba_rot @ depth2rgba_sec_trans + sec2main_rgba_trans)
        ) + reference_data["bgra2depth_trans"]

        trans_matrix = np.eye(4, dtype=np.float32)
        trans_matrix[:3, :3] = sec2main_depth_rot
        trans_matrix[:3, 3] = sec2main_depth_trans
        trans_matrices[i] = trans_matrix

    return trans_matrices
