import pathlib

import numpy as np

import fast_body_tracker as fbt


def main(base_dir: pathlib.Path | str):
    base_dir = pathlib.Path(base_dir)
    trans_matrices_path = base_dir / "trans_matrices.npz"
    with np.load(trans_matrices_path) as data:
        trans_matrices = {
            int(device_id): np.asarray(matrix, dtype=np.float32)
            for device_id, matrix in data.items()
        }
    fbt.check_calibration(trans_matrices=trans_matrices)


if __name__ == "__main__":
    base_dir = pathlib.Path("../data/")
    main(base_dir=base_dir)
