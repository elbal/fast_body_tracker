import pathlib
import h5py

import fast_body_tracker as fbt


def main(h5_file_path: pathlib.Path | str):
    h5_file_path = pathlib.Path(h5_file_path)

    with h5py.File(h5_file_path, "r") as h5file:
        joints_group = h5file["joints"]
        ts_group = h5file["ts"]
        ts_dict = {key: ts_group[key][:] for key in ts_group.keys()}
        joints_dict = {
            body_name: {
                key: joints_group[body_name][key][:]
                for key in joints_group[body_name].keys()
            }
            for body_name in joints_group.keys()
        }

    viewer = fbt.BodyVisualizer(ts_dict=ts_dict, joints_dict=joints_dict)
    viewer.run()


if __name__ == "__main__":
    main(pathlib.Path(__file__).resolve().parents[1] / "data" / "test_run" / "body.h5")
