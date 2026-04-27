from pathlib import Path

import cv2
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
import numpy as np


def save_charuco_board(board_ids: int | list[int] = 0, path: str | Path = "."):
    a3_width_mm = 297
    a3_height_mm = 420
    dpi = 300
    pixel_w = round(a3_width_mm / 25.4 * dpi)
    pixel_h = round(a3_height_mm / 25.4 * dpi)

    n_squares_w = 3
    n_squares_h = 3
    square_length_mm = 94.0
    board_pixel_w = round(n_squares_w * square_length_mm / 25.4 * dpi)
    board_pixel_h = round(n_squares_h * square_length_mm / 25.4 * dpi)
    marker_length_mm = 94.0 * 0.76
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

    board_ids = [board_ids] if isinstance(board_ids, int) else board_ids
    n_marker = (n_squares_w * n_squares_h) // 2
    max_board_id = 1000 // n_marker - 1
    if min(board_ids) < 0 or max(board_ids) > max_board_id:
        raise ValueError("Board ID is out of range for DICT_5X5_1000.")

    for board_id in board_ids:
        marker_start = board_id * n_marker
        marker_ids = np.arange(marker_start, marker_start + n_marker, dtype=np.int32)
        board = cv2.aruco.CharucoBoard(
            size=(n_squares_w, n_squares_h),
            squareLength=square_length_mm,
            markerLength=marker_length_mm,
            dictionary=aruco_dict,
            ids=marker_ids,
        )
        image = board.generateImage(
            outSize=(board_pixel_w, board_pixel_h), marginSize=0
        )

        canvas = np.ones((pixel_h, pixel_w), dtype=np.uint8) * 255
        h, w = image.shape[:2]
        y0 = (pixel_h - h) // 2
        x0 = (pixel_w - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = image

        ids_str = "_".join(str(marker_id) for marker_id in marker_ids)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5.0
        thickness = 12
        (text_w, text_h), baseline = cv2.getTextSize(
            ids_str, font, font_scale, thickness
        )
        text_x = (pixel_w - text_w) // 2
        text_y = max(text_h + 40, (y0 - baseline) // 2 + text_h // 2)
        cv2.putText(
            canvas,
            ids_str,
            (text_x, text_y),
            font,
            font_scale,
            0,
            thickness,
            cv2.LINE_AA,
        )

        figure = Figure(
            figsize=(a3_width_mm / 25.4, a3_height_mm / 25.4),
            dpi=dpi,
            frameon=False,
        )
        FigureCanvasPdf(figure)
        axes = figure.add_axes([0, 0, 1, 1])
        axes.imshow(canvas, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes.set_axis_off()

        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"charuco_board_A3_3x3_id_{ids_str}.pdf"
        figure.savefig(
            output_file,
            format="pdf",
            bbox_inches=None,
            pad_inches=0,
        )
