import numpy as np

import fast_body_tracker as fbt


def main():
    transformation_matrices = fbt.external_calibration([0, 1, 2])
    save_data = {str(k): v for k, v in transformation_matrices.items()}
    np.savez("../data/transformation_matrices.npz", **save_data)


if __name__ == "__main__":
    main()
