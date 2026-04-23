import h5py
import json
import collections
from pathlib import Path

JOINT_NAMES = [
    "PELVIS",
    "SPINE_NAVEL",
    "SPINE_CHEST",
    "NECK",
    "CLAVICLE_LEFT",
    "SHOULDER_LEFT",
    "ELBOW_LEFT",
    "WRIST_LEFT",
    "HAND_LEFT",
    "HANDTIP_LEFT",
    "THUMB_LEFT",
    "CLAVICLE_RIGHT",
    "SHOULDER_RIGHT",
    "ELBOW_RIGHT",
    "WRIST_RIGHT",
    "HAND_RIGHT",
    "HANDTIP_RIGHT",
    "THUMB_RIGHT",
    "HIP_LEFT",
    "KNEE_LEFT",
    "ANKLE_LEFT",
    "FOOT_LEFT",
    "HIP_RIGHT",
    "KNEE_RIGHT",
    "ANKLE_RIGHT",
    "FOOT_RIGHT",
    "HEAD",
    "NOSE",
    "EYE_LEFT",
    "EAR_LEFT",
    "EYE_RIGHT",
    "EAR_RIGHT",
]

BONE_LIST = [
    ["SPINE_NAVEL", "PELVIS"],
    ["SPINE_CHEST", "SPINE_NAVEL"],
    ["NECK", "SPINE_CHEST"],
    ["HEAD", "NECK"],
    ["SPINE_CHEST", "CLAVICLE_LEFT"],
    ["CLAVICLE_LEFT", "SHOULDER_LEFT"],
    ["SHOULDER_LEFT", "ELBOW_LEFT"],
    ["ELBOW_LEFT", "WRIST_LEFT"],
    ["WRIST_LEFT", "THUMB_LEFT"],
    ["THUMB_LEFT", "HAND_LEFT"],
    ["WRIST_LEFT", "HANDTIP_LEFT"],
    ["SPINE_CHEST", "CLAVICLE_RIGHT"],
    ["CLAVICLE_RIGHT", "SHOULDER_RIGHT"],
    ["SHOULDER_RIGHT", "ELBOW_RIGHT"],
    ["ELBOW_RIGHT", "WRIST_RIGHT"],
    ["WRIST_RIGHT", "HAND_RIGHT"],
    ["HAND_RIGHT", "THUMB_RIGHT"],
    ["WRIST_RIGHT", "HANDTIP_RIGHT"],
    ["PELVIS", "HIP_LEFT"],
    ["HIP_LEFT", "KNEE_LEFT"],
    ["KNEE_LEFT", "ANKLE_LEFT"],
    ["ANKLE_LEFT", "FOOT_LEFT"],
    ["PELVIS", "HIP_RIGHT"],
    ["HIP_RIGHT", "KNEE_RIGHT"],
    ["KNEE_RIGHT", "ANKLE_RIGHT"],
    ["ANKLE_RIGHT", "FOOT_RIGHT"],
    ["NOSE", "HEAD"],
    ["EYE_LEFT", "NOSE"],
    ["NOSE", "EYE_LEFT"],
    ["EYE_LEFT", "EAR_LEFT"],
    ["NOSE", "EYE_RIGHT"],
    ["EYE_RIGHT", "EAR_RIGHT"],
]


DEFAULT_OUTPUT_DIR = Path("C:/Users/eliab/Python/postura/data")


def process_h5_to_json(h5_file_path: str, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
    path = Path(h5_file_path)
    folder_timestamp = path.parents[0].name
    folder_ex = path.parents[1].name
    output_dir = Path(output_dir)
    with h5py.File(h5_file_path, "r") as f:
        if "joints" in f and "ts" in f:
            _process_body_saver_h5(f, folder_ex, folder_timestamp, output_dir)
            return

        devices = collections.defaultdict(list)
        for key in f.keys():
            if key.startswith("device_"):
                device_name, body_str = key.rsplit("_", 1)
                body_id = int(body_str.replace("body", "")) + 1
                devices[device_name].append((key, body_id))

        for device_name, body_keys in devices.items():
            frames_dict = {}
            for key, body_id in body_keys:
                grp = f[key]

                frame_indices = grp["frame_idx"][:].tolist()
                timestamps = grp["ts"][:].tolist()
                positions = grp["positions"][:].tolist()

                for i in range(len(frame_indices)):
                    f_idx = frame_indices[i]
                    if f_idx not in frames_dict:
                        frames_dict[f_idx] = {
                            "bodies": [],
                            "frame_id": f_idx,
                            "num_bodies": 0,
                            "timestamp_usec": timestamps[i],
                        }

                    frames_dict[f_idx]["bodies"].append(
                        {"body_id": body_id, "joint_positions": positions[i]}
                    )
                    frames_dict[f_idx]["num_bodies"] += 1
            sorted_frames = [frames_dict[k] for k in sorted(frames_dict.keys())]
            output_data = {
                "bone_list": BONE_LIST,
                "frames": sorted_frames,
                "joint_names": JOINT_NAMES,
            }

            output_filename = f"{folder_ex}_{folder_timestamp}_{device_name}.json"
            _write_output_json(output_dir, output_filename, output_data)


def _process_body_saver_h5(
    h5file: h5py.File, folder_ex: str, folder_timestamp: str, output_dir: Path
):
    joints_group = h5file["joints"]
    ts_group = h5file["ts"]

    for device_name in sorted(joints_group.keys(), key=_device_sort_key):
        frame_timestamps = ts_group[device_name]["ts"][:].tolist()
        frames_dict = {}
        device_group = joints_group[device_name]

        for body_name in sorted(device_group.keys(), key=_body_sort_key):
            body_id = int(body_name.replace("body_", "")) + 1
            body_group = device_group[body_name]

            frame_indices = body_group["frame_idx"][:].tolist()
            positions = body_group["positions"][:].tolist()

            for frame_idx, joint_positions in zip(frame_indices, positions):
                if frame_idx not in frames_dict:
                    timestamp_usec = None
                    if 0 <= frame_idx < len(frame_timestamps):
                        timestamp_usec = frame_timestamps[frame_idx]
                    frames_dict[frame_idx] = {
                        "bodies": [],
                        "frame_id": frame_idx,
                        "num_bodies": 0,
                        "timestamp_usec": timestamp_usec,
                    }

                frames_dict[frame_idx]["bodies"].append(
                    {
                        "body_id": body_id,
                        "joint_positions": joint_positions,
                    }
                )
                frames_dict[frame_idx]["num_bodies"] += 1

        sorted_frames = [frames_dict[k] for k in sorted(frames_dict.keys())]
        output_data = {
            "bone_list": BONE_LIST,
            "frames": sorted_frames,
            "joint_names": JOINT_NAMES,
        }

        output_filename = f"{folder_ex}_{folder_timestamp}_{device_name}.json"
        _write_output_json(output_dir, output_filename, output_data)


def _write_output_json(output_dir: Path, output_filename: str, output_data: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    with open(output_path, "w") as out_file:
        json.dump(output_data, out_file, indent=4)


def _device_sort_key(device_name: str) -> int:
    return int(device_name.replace("device_", ""))


def _body_sort_key(body_name: str) -> int:
    return int(body_name.replace("body_", ""))


if __name__ == "__main__":
    process_h5_to_json(
        "C:/Users/eliab/Python/fast_body_tracker/data/test/body.h5", DEFAULT_OUTPUT_DIR
    )
