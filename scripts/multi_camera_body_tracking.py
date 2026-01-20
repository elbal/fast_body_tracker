import queue
import numpy as np
from numpy import typing as npt
import threading
import pathlib
from datetime import datetime
from typing import Tuple

import fast_body_tracker as fbt

def main(base_dir: pathlib.Path | str, n_bodies: int=1):
    trans_matrices_path = base_dir / "trans_matrices.npz"
    with np.load(trans_matrices_path) as data:
        trans_matrices = {int(k): v for k, v in data.items()}
    fbt.default_pipeline(
        base_dir=base_dir, trans_matrices=trans_matrices, sync=True,
        n_bodies=n_bodies)

if __name__ == "__main__":
    base_dir = pathlib.Path("../data/")
    main(base_dir=base_dir, n_bodies=1)
