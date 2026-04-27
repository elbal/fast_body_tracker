import pathlib

import fast_body_tracker as fbt


def main(board_ids: int | list[int], base_dir: pathlib.Path | str):
    base_dir = pathlib.Path(base_dir)
    fbt.save_charuco_board(board_ids=board_ids, path=base_dir)


if __name__ == "__main__":
    base_dir = pathlib.Path(__file__).resolve().parents[1] / "data"
    main(board_ids=[0, 1], base_dir=base_dir)
