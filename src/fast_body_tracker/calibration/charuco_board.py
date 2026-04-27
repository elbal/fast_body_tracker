import cv2
import numpy as np


def save_charuco_board(path: str, board_id: int | list[int] = 0):
    a3_width_mm = 297
    a3_height_mm = 420
    dpi = 300

    n_squares_w = 3
    n_squares_h = 3
    square_length_mm = 94.0
    marker_length_mm = 94.0 * 0.76
    marker_count = (n_squares_w * n_squares_h) // 2

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    pixel_w = round(a3_width_mm / 25.4 * dpi)
    pixel_h = round(a3_height_mm / 25.4 * dpi)
    board_pixel_w = round(n_squares_w * square_length_mm / 25.4 * dpi)
    board_pixel_h = round(n_squares_h * square_length_mm / 25.4 * dpi)

    board_ids = [board_id] if isinstance(board_id, int) else board_id
    max_board_id = 1000 // marker_count - 1
    if min(board_ids) < 0 or max(board_ids) > max_board_id:
        raise ValueError("Board ID is out of range for DICT_5X5_1000.")

    for current_board_id in board_ids:
        marker_start = current_board_id * marker_count

        marker_ids = np.arange(
            marker_start, marker_start + marker_count, dtype=np.int32
        )
        board = cv2.aruco.CharucoBoard(
            size=(n_squares_w, n_squares_h),
            squareLength=square_length_mm,
            markerLength=marker_length_mm,
            dictionary=aruco_dict,
            ids=marker_ids,
        )

        image = board.generateImage(
            outSize=(board_pixel_w, board_pixel_h), marginSize=0, borderBits=1
        )

        canvas = np.ones((pixel_h, pixel_w), dtype=np.uint8) * 255
        h, w = image.shape[:2]
        y0 = (pixel_h - h) // 2
        x0 = (pixel_w - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = image

        marker_ids_str = "_".join(str(marker_id) for marker_id in marker_ids)
        output_file = f"{path}charuco_board_A3_3x3_id_{marker_ids_str}.png"
        cv2.imwrite(output_file, canvas)
