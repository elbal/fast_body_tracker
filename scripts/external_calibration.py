import numpy as np
import pathlib

import fast_body_tracker as fbt


def main(base_dir: pathlib.Path | str, n_devices: int=1):
    trans_matrices_path = base_dir / "trans_matrices.npz"
    trans_matrices = fbt.external_calibration(n_devices=n_devices)
    save_data = {str(k): v for k, v in trans_matrices.items()}
    np.savez(trans_matrices_path , **save_data)


if __name__ == "__main__":
    base_dir = pathlib.Path("../data/")
    main(base_dir=base_dir, n_devices=2)
