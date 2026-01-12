import cv2
import numpy as np


def save_charuco_board(path: str):
    a3_width_mm = 297
    a3_height_mm = 420
    dpi = 300

    n_squares_w = 3
    n_squares_h = 3
    square_length_mm = 94.0
    marker_length_mm = 94.0 * 0.76

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard(
        size=(n_squares_w, n_squares_h), squareLength=square_length_mm,
        markerLength=marker_length_mm, dictionary=aruco_dict)

    pixel_w = int(a3_width_mm / 25.4 * dpi)
    pixel_h = int(a3_height_mm / 25.4 * dpi)

    image = board.generateImage(
        outSize=(pixel_w - 100, pixel_h - 100), marginSize=20, borderBits=1)

    canvas = np.ones((pixel_h, pixel_w), dtype=np.uint8) * 255
    h, w = image.shape[:2]
    y0 = (pixel_h - h) // 2
    x0 = (pixel_w - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = image

    output_file = f"{path}charuco_board_A3_3x3.png"
    cv2.imwrite(output_file, canvas)
