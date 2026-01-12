import cv2
import numpy as np


def save_charuco_board(path: str):
    """
    Save a printable (3x3) A3 ChArUco board image.

    Parameters
    ----------
    path : str
        Path to save the ChArUco board image.
    """

    # A3 dimensions (mm).
    a3_width_mm = 297
    a3_height_mm = 420
    dpi = 300
    # 3x3 configuration.
    n_squares_w = 3
    n_squares_h = 3
    square_length_mm = 125  # large squares.
    marker_length_mm = 95  # large markers.

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard(
        size=(n_squares_w, n_squares_h), squareLength=square_length_mm,
        markerLength=marker_length_mm, dictionary=aruco_dict)

    pixel_w = int(a3_width_mm / 25.4 * dpi)
    pixel_h = int(a3_height_mm / 25.4 * dpi)
    pixel_margin = 20
    image = board.generateImage(
        outSize=(pixel_w - 2 * pixel_margin, pixel_h - 2 * pixel_margin),
        marginSize=pixel_margin, borderBits=1)
    # Center the board on a white canvas.
    canvas = np.ones((pixel_h, pixel_w), dtype=image.dtype) * 255
    h, w = image.shape[:2]
    y0 = (pixel_h-h) // 2
    x0 = (pixel_w-w) // 2
    canvas[y0:y0+h, x0:x0+w] = image

    cv2.imwrite(path + "charuco_board_A3_3x3.png", canvas)


if __name__ == "__main__":
    save_charuco_board("../../data/charuco_boards/")
