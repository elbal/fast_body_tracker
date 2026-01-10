import numpy as np
from time import perf_counter
from vispy import scene
from vispy.scene.visuals import Markers, Text, Line, GridLines


class PointCloudVisualizer:
    def __init__(self):
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=True, title="Point cloud", vsync=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45, distance=2000, up="-y")

        self.scatter = Markers(parent=self.view.scene)

        self._fps_text = Text(
            "FPS: 0", color="white", font_size=12, parent=self.canvas.scene,
            anchor_x="left", anchor_y="top",
            pos=(10, self.canvas.size[1] - 10))
        self._frame_count = 0
        self._start_time = perf_counter()

        self._center_camera_flag = True

    def update(self, point_cloud, bgra_image=None):
        self._fps_text.pos = (10, self.canvas.size[1] - 10)

        valid_mask = point_cloud[:, 2] != 0
        points = point_cloud[valid_mask]

        if self._center_camera_flag:
            self.view.camera.center = np.median(points, axis=0)
            self._center_camera_flag = False

        if bgra_image is not None:
            colors = bgra_image.reshape(-1, 4)[valid_mask]
            colors_filtered = colors[:, [2, 1, 0]].astype(np.float32) / 255.0
        else:
            colors_filtered = (1, 1, 1)

        self.scatter.set_data(
            pos=points, face_color=colors_filtered, size=2, edge_width=0)

        self.canvas.app.process_events()
        self._update_fps()

    def _update_fps(self):
        self._frame_count += 1
        end_time = perf_counter()
        elapsed_time = end_time - self._start_time
        if elapsed_time >= 1.0:
            self._fps_text.text = f"FPS: {self._frame_count/elapsed_time:.2f}"
            self._frame_count = 0
            self._start_time = perf_counter()


class IMUVisualizer:
    def __init__(self, max_samples=400):
        self.max_samples = max_samples
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=True, title="IMU data", vsync=False)
        grid = self.canvas.central_widget.add_grid()

        colors = [(0.98, 0.90, 0.07), (0.47, 0.51, 0.53), (0.22, 0.38, 0.58)]
        labels = ["X", "Y", "Z"]

        self.lines = {"accel": [], "gyro": []}
        configs = [("m/s²", (-20, 20), "accel", 0),
                   ("rad/s", (-5, 5), "gyro", 1)]

        for label, y_range, key, row in configs:
            yaxis = scene.AxisWidget(
                orientation="left", axis_label=label, font_size=8)
            yaxis.width_max = 60
            grid.add_widget(yaxis, row=row, col=0)

            view = grid.add_view(row=row, col=1, border_color="white")
            view.camera = "panzoom"
            view.camera.set_range(x=(0, max_samples), y=y_range)
            yaxis.link_view(view)
            GridLines(parent=view.scene, color=(0.5, 0.5, 0.5, 0.5))

            for i in range(3):
                self.lines[key].append(
                    Line(parent=view.scene, color=colors[i], width=2))

        self.x_axis = np.arange(max_samples, dtype=np.float32)
        self.accel_buf = np.zeros((max_samples, 3), dtype=np.float32)
        self.gyro_buf = np.zeros((max_samples, 3), dtype=np.float32)

        self._fps_text = Text(
            "FPS: 0", color="white", font_size=12, parent=self.canvas.scene,
            anchor_x="right", anchor_y="top") # Changed to anchor_x="right"

        self.legend_items = []
        for i, (label, color) in enumerate(zip(labels, colors)):
            t = Text(
                label, color=color, font_size=14, parent=self.canvas.scene,
                anchor_x="right", anchor_y="top")
            self.legend_items.append(t)

        self._frame_count = 0
        self._start_time = perf_counter()

    def update(self, imu_samples):
        w, h = self.canvas.size
        self._fps_text.pos = (w - 10, h - 10)
        for i, t in enumerate(self.legend_items):
            t.pos = (w - 10, 30 + i*25)

        num_new = len(imu_samples)
        self.accel_buf = np.roll(self.accel_buf, -num_new, axis=0)
        self.gyro_buf = np.roll(self.gyro_buf, -num_new, axis=0)

        self.accel_buf[-num_new:] = [s.acc for s in imu_samples]
        self.gyro_buf[-num_new:] = [s.gyro for s in imu_samples]

        for i in range(3):
            self.lines["accel"][i].set_data(
                pos=np.column_stack((self.x_axis, self.accel_buf[:, i])))
            self.lines["gyro"][i].set_data(
                pos=np.column_stack((self.x_axis, self.gyro_buf[:, i])))

        self.canvas.app.process_events()
        self._update_fps()

    def _update_fps(self):
        self._frame_count += 1
        end_time = perf_counter()
        elapsed_time = end_time - self._start_time
        if elapsed_time >= 1.0:
            self._fps_text.text = f"FPS: {self._frame_count/elapsed_time:.2f}"
            self._frame_count = 0
            self._start_time = perf_counter()
