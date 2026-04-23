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


def process_h5_to_json(h5_file_path: str):
    path = Path(h5_file_path)
    folder_timestamp = path.parents[0].name
    folder_ex = path.parents[1].name
    with h5py.File(h5_file_path, "r") as f:
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

            out_dir = Path(folder_ex) / folder_timestamp
            out_dir.mkdir(parents=True, exist_ok=True)
            output_filename = f"{folder_ex}_{folder_timestamp}_{device_name}.json"
            output_path = out_dir / output_filename
            with open(output_path, "w") as out_file:
                json.dump(output_data, out_file, indent=4)
