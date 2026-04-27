import pathlib

import fast_body_tracker as fbt


def main(board_id: int | list[int], base_dir: pathlib.Path | str):
    base_dir = pathlib.Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    fbt.save_charuco_board(path=str(base_dir) + "/", board_id=board_id)


if __name__ == "__main__":
    base_dir = pathlib.Path("../data/")
    main(board_id=[0, 1], base_dir=base_dir)
