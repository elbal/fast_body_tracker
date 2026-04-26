import numpy as np
import numpy.typing as npt
from vispy import app, scene
from vispy.color import get_colormap
from vispy.scene.visuals import Line, Markers, Mesh, Text

from ..k4abt.kabt_const import K4ABT_SEGMENT_PAIRS


class BodyVisualizer:
    def __init__(
        self,
        ts_dict: dict[str, npt.NDArray[np.int64]],
        joints_dict: dict[
            str, dict[str, npt.NDArray[np.int64] | npt.NDArray[np.float32]]
        ],
    ):
        self.frame_indices = np.asarray(ts_dict["frame_idx"], dtype=np.int64)
        self.timestamps_usec = np.asarray(ts_dict["ts"], dtype=np.int64)
        if self.frame_indices.size == 0:
            raise RuntimeError("No frames found.")

        self.body_tracks = []
        all_positions_parts = []
        unique_tags = set()
        self.segment_pairs = np.asarray(K4ABT_SEGMENT_PAIRS, dtype=np.int32)
        self.empty_segments = np.empty((0, 2), dtype=np.int32)

        for body_name in sorted(
            joints_dict,
            key=lambda current_body_name: int(current_body_name.replace("body_", "")),
        ):
            body_dict = joints_dict[body_name]
            body_frame_indices = np.asarray(body_dict["frame_idx"], dtype=np.int64)
            body_tags = np.asarray(body_dict["tags"], dtype=np.int64)
            body_positions = np.asarray(body_dict["positions"], dtype=np.float32)
            if body_frame_indices.size == 0 or body_positions.size == 0:
                continue

            self.body_tracks.append(
                {
                    "name": body_name,
                    "frame_idx": body_frame_indices,
                    "tags": body_tags,
                    "positions": body_positions,
                }
            )
            all_positions_parts.append(body_positions.reshape(-1, 3))
            unique_tags.update(int(tag) for tag in body_tags)

        if not all_positions_parts:
            raise RuntimeError("No body positions found.")

        self.n_frames = int(self.frame_indices.size)
        self.start_ts_usec = int(self.timestamps_usec[0])
        self.current_frame = 0
        self.paused = False
        self.body_cursors = np.full(len(self.body_tracks), -1, dtype=np.int64)
        self._sync_body_cursors(self.current_frame)

        app.use_app("pyside6")

        all_positions = np.concatenate(all_positions_parts, axis=0)
        min_xyz = all_positions.min(axis=0)
        max_xyz = all_positions.max(axis=0)
        center = (min_xyz + max_xyz) / 2
        radius = np.linalg.norm(max_xyz - min_xyz) / 2

        self.canvas = scene.SceneCanvas(
            keys="interactive", show=True, bgcolor="white", title="Body visualizer"
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45, azimuth=30, elevation=30, distance=radius, up="z"
        )
        self.view.camera.center = center

        xy_span = max(
            float(max_xyz[0] - min_xyz[0]), float(max_xyz[1] - min_xyz[1]), 1.0
        )
        floor_pad = 0.1 * xy_span
        x_min = min(float(min_xyz[0]), 0.0) - floor_pad
        x_max = max(float(max_xyz[0]), 0.0) + floor_pad
        y_min = min(float(min_xyz[1]), 0.0) - floor_pad
        y_max = max(float(max_xyz[1]), 0.0) + floor_pad
        floor_vertices = np.array(
            [
                [x_min, y_min, 0.0],
                [x_max, y_min, 0.0],
                [x_max, y_max, 0.0],
                [x_min, y_max, 0.0],
            ],
            dtype=np.float32,
        )
        floor_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        self.floor = Mesh(
            vertices=floor_vertices,
            faces=floor_faces,
            color=(0.85, 0.85, 0.85, 0.35),
            parent=self.view.scene,
        )
        self.floor.set_gl_state("translucent", depth_test=True)

        self.frame_text = Text(
            text="",
            color="black",
            font_size=12,
            parent=self.canvas.scene,
            anchor_x="left",
            anchor_y="top",
        )
        self.frame_text.transform = scene.transforms.STTransform(translate=(10, 30))

        self.ts_text = Text(
            text="",
            color="black",
            font_size=12,
            parent=self.canvas.scene,
            anchor_x="left",
            anchor_y="top",
        )
        self.ts_text.transform = scene.transforms.STTransform(translate=(10, 60))

        cmap = get_colormap("viridis")
        n_tags = max(len(unique_tags), 1)
        self.tag_colors = {
            tag: cmap.map(np.array([i / n_tags], dtype=np.float32))[0]
            for i, tag in enumerate(sorted(unique_tags))
        }

        max_bodies = len(self.body_tracks)
        self.body_markers = []
        self.body_segments = []
        self.tag_texts = []
        for i in range(max_bodies):
            marker = Markers(parent=self.view.scene)
            self.body_markers.append(marker)

            segment_line = Line(parent=self.view.scene)
            self.body_segments.append(segment_line)

            tag_text = Text(
                text="",
                color="black",
                font_size=12,
                parent=self.canvas.scene,
                anchor_x="right",
                anchor_y="top",
            )
            tag_text.transform = scene.transforms.STTransform(
                translate=(790, 30 + i * 20)
            )
            self.tag_texts.append(tag_text)

        Markers(
            pos=np.array([[0, 0, 0]], dtype=np.float32),
            parent=self.view.scene,
            face_color="black",
            size=5,
        )

        self.canvas.events.key_press.connect(self.on_key)
        self.canvas.events.resize.connect(self.on_resize)
        self.timer = app.Timer(interval=1 / 30, connect=self.on_timer, start=False)

    def _sync_body_cursors(self, ts_index: int):
        target_frame_idx = int(self.frame_indices[ts_index])
        for track_idx, track in enumerate(self.body_tracks):
            self.body_cursors[track_idx] = (
                np.searchsorted(track["frame_idx"], target_frame_idx, side="right") - 1
            )

    def _advance_to_frame(self, next_ts_index: int):
        if next_ts_index == self.current_frame:
            return

        target_frame_idx = int(self.frame_indices[next_ts_index])
        moving_forward = next_ts_index > self.current_frame

        for track_idx, track in enumerate(self.body_tracks):
            cursor = int(self.body_cursors[track_idx])
            frame_idx = track["frame_idx"]

            if moving_forward:
                while (
                    cursor + 1 < len(frame_idx)
                    and frame_idx[cursor + 1] <= target_frame_idx
                ):
                    cursor += 1
            else:
                while cursor >= 0 and frame_idx[cursor] > target_frame_idx:
                    cursor -= 1

            self.body_cursors[track_idx] = cursor

        self.current_frame = next_ts_index

    def _current_bodies(self) -> list[dict[str, int | npt.NDArray[np.float32]]]:
        target_frame_idx = int(self.frame_indices[self.current_frame])
        bodies = []

        for track_idx, track in enumerate(self.body_tracks):
            cursor = int(self.body_cursors[track_idx])
            if cursor < 0 or track["frame_idx"][cursor] != target_frame_idx:
                continue

            bodies.append(
                {
                    "tag": int(track["tags"][cursor]),
                    "positions": track["positions"][cursor],
                }
            )

        bodies.sort(key=lambda body: body["tag"])
        return bodies

    def update_visuals(self):
        frame_idx = int(self.frame_indices[self.current_frame])
        timestamp_usec = int(self.timestamps_usec[self.current_frame])
        bodies = self._current_bodies()

        self.frame_text.text = f"Frame: {frame_idx}/{self.n_frames}"

        relative_ts = timestamp_usec - self.start_ts_usec
        total_seconds = relative_ts * 1e-6
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        self.ts_text.text = f"TS: {minutes:02d}:{seconds:05.2f}"

        for i, marker in enumerate(self.body_markers):
            if i < len(bodies):
                body = bodies[i]
                color = self.tag_colors[body["tag"]]
                positions = body["positions"]
                marker.set_data(
                    pos=positions,
                    face_color=color,
                    size=5,
                    edge_width=0,
                )
                self.body_segments[i].set_data(
                    pos=positions,
                    color=color,
                    width=2,
                    connect=self.segment_pairs,
                )
                self.tag_texts[i].text = f"Tag {body['tag']}"
                self.tag_texts[i].color = color
            else:
                marker.set_data(pos=np.empty((0, 3), dtype=np.float32))
                self.body_segments[i].set_data(
                    pos=np.empty((0, 3), dtype=np.float32),
                    connect=self.empty_segments,
                )
                self.tag_texts[i].text = ""

    def on_resize(self, event):
        width = self.canvas.size[0]
        for i, text in enumerate(self.tag_texts):
            text.transform = scene.transforms.STTransform(
                translate=(width - 10, 30 + i * 20)
            )

    def on_timer(self, event):
        if not self.paused:
            self._advance_to_frame((self.current_frame + 1) % self.n_frames)
            self.update_visuals()

    def on_key(self, event):
        if event.key == "Space":
            self.paused = not self.paused
        elif event.key == "Left":
            self.paused = True
            self._advance_to_frame(max(0, self.current_frame - 1))
            self.update_visuals()
        elif event.key == "Right":
            self.paused = True
            self._advance_to_frame(min(self.n_frames - 1, self.current_frame + 1))
            self.update_visuals()

    def run(self):
        self.update_visuals()
        self.timer.start()
        app.run()
