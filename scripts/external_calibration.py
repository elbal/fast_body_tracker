import numpy as np
import pathlib

import fast_body_tracker as fbt


def main(base_dir: pathlib.Path | str):
    base_dir = pathlib.Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    trans_matrices = fbt.external_calibration()
    trans_matrices = {str(k): v for k, v in trans_matrices.items()}

    trans_matrices_path = base_dir / "trans_matrices.npz"
    np.savez(trans_matrices_path, **trans_matrices)


if __name__ == "__main__":
    base_dir = pathlib.Path("../data/")
    main(base_dir=base_dir)
