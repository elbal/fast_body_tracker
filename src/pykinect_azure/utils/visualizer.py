import numpy as np
from numpy import typing as npt
from time import perf_counter
from vispy import scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Markers, Text, Line, GridLines
from vispy.scene.cameras import TurntableCamera


class PointCloudVisualizer:
	def __init__(self):
		self.color_lut = np.arange(256, dtype=np.float32) / 255.0
		self.canvas = SceneCanvas(
			keys="interactive", show=True, title="Point cloud")
		self.view = self.canvas.central_widget.add_view()
		self.view.camera = TurntableCamera(
			fov=45, distance=3000, up="-y", elevation=30, azimuth=0,
			flip=(False, True, False))
		self.view.camera.set_range(
			x=(-2000, 2000), y=(-2000, 2000), z=(-2000, 2000))

		self.flip_vector = np.array([1, -1, -1], dtype=np.int16)
		self.scatter = Markers(parent=self.view.scene)

		self.fps_text = Text(
			"FPS: 0", color="white", font_size=12, bold=True,
			parent=self.canvas.scene, anchor_x="left", anchor_y="top",
			pos=(10, self.canvas.size[1] - 10))
		self._last_time = perf_counter()
		self._frames = 0

		self.first_valid_frame = True
		self.initial_center = np.array([0, 0, 1500], dtype=np.float32)

	def __call__(
			self, points: npt.NDArray[np.int16],
			bgra_data: npt.NDArray[np.uint8] =None):
		self.update(points, bgra_data)

	def update(
			self, point_cloud: npt.NDArray[np.int16],
			bgra_image: npt.NDArray[np.uint8] = None):
		self._update_fps()
		if point_cloud is None or len(point_cloud) == 0:
			return
		valid_mask = point_cloud[:, 2] != 0
		points_filtered = point_cloud[valid_mask]
		if points_filtered.size == 0:
			return
		points_filtered *= self.flip_vector

		if bgra_image is not None:
			bgra_flat = bgra_image.reshape(-1, 4)
			colors_subset = bgra_flat[valid_mask]
			colors_filtered = self.color_lut[colors_subset[:, [2, 1, 0]]]
		else:
			colors_filtered = (1, 1, 1)

		if self.first_valid_frame:
			self.initial_center = np.median(points_filtered, axis=0)
			self.view.camera.center = tuple(self.initial_center)
			q10 = np.percentile(points_filtered, 10, axis=0)
			q90 = np.percentile(points_filtered, 90, axis=0)
			robust_radius = np.linalg.norm(q90-q10) / 2
			if robust_radius > 500:
				self.view.camera.distance = robust_radius * 2.5
			self.first_valid_frame = False

		self.scatter.set_data(
			pos=points_filtered, face_color=colors_filtered, size=2,
			edge_width=0)
		self.canvas.app.process_events()

	def _update_fps(self):
		self._frames += 1
		now = perf_counter()
		dt = now - self._last_time
		if dt >= 1.0:
			fps = self._frames / dt
			self.fps_text.text = f"FPS: {fps:.1f}"
			self.fps_text.pos = (10, self.canvas.size[1] - 10)
			self._frames = 0
			self._last_time = now


class IMUVisualizer:
	def __init__(self, max_samples=400):
		self.max_samples = max_samples
		self.canvas = scene.SceneCanvas(
			keys="interactive", show=True, title="IMU data")
		self.grid_layout = self.canvas.central_widget.add_grid()

		self.colors = [
			(0.98, 0.90, 0.07), (0.47, 0.51, 0.53), (0.22, 0.38, 0.58)]
		labels = ["X", "Y", "Z"]

		# Accelerometer.
		self.yaxis_accel = scene.AxisWidget(
			orientation="left", axis_label="m/s²", font_size=8)
		self.yaxis_accel.width_max = 60
		self.grid_layout.add_widget(self.yaxis_accel, row=0, col=0)
		self.view_accel = self.grid_layout.add_view(
			row=0, col=1, border_color="white")
		self.view_accel.camera = "panzoom"
		self.view_accel.camera.set_range(x=(0, self.max_samples), y=(-20, 20))
		self.yaxis_accel.link_view(self.view_accel)
		self.grid_accel = GridLines(
			parent=self.view_accel.scene, color=(0.5, 0.5, 0.5, 0.5))

		# Gyroscope.
		self.yaxis_gyro = scene.AxisWidget(
			orientation="left", axis_label="rad/s", font_size=8)
		self.yaxis_gyro.width_max = 60
		self.grid_layout.add_widget(self.yaxis_gyro, row=1, col=0)
		self.view_gyro = self.grid_layout.add_view(
			row=1, col=1, border_color="white")
		self.view_gyro.camera = "panzoom"
		self.view_gyro.camera.set_range(x=(0, self.max_samples), y=(-5, 5))
		self.yaxis_gyro.link_view(self.view_gyro)
		self.grid_gyro = GridLines(
			parent=self.view_gyro.scene, color=(0.5, 0.5, 0.5, 0.5))

		self.x_axis = np.arange(self.max_samples, dtype=np.float32)

		self.accel_buffer = np.zeros(
			(self.max_samples, 3), dtype=np.float32)
		self.gyro_buffer = np.zeros(
			(self.max_samples, 3), dtype=np.float32)

		self.lines_a = [
			Line(parent=self.view_accel.scene, color=self.colors[i], width=2)
			for i in range(3)]
		self.lines_g = [
			Line(parent=self.view_gyro.scene, color=self.colors[i], width=2)
			for i in range(3)]

		self.fps_text = Text(
			"FPS: 0", color="white", font_size=12, bold=True,
			parent=self.canvas.scene, anchor_x="left", anchor_y="top",
			pos=(10, self.canvas.size[1] - 10))

		self.legend_texts = []
		for i in range(3):
			t = Text(
				labels[i], color=self.colors[i], font_size=14, bold=True,
				parent=self.canvas.scene, anchor_x="right", anchor_y="top")
			self.legend_texts.append(t)
		self._update_legend_pos()

		self._last_time = perf_counter()
		self._frames = 0

	def _update_legend_pos(self):
		w, h = self.canvas.size
		for i, t in enumerate(self.legend_texts):
			t.pos = (w - 50, 30 + i*25)

	def __call__(self, imu_samples):
		self.update(imu_samples)

	def update(self, imu_samples):
		self._update_fps()
		self._update_legend_pos()
		if not imu_samples:
			return

		num_new = len(imu_samples)
		self.accel_buffer = np.roll(self.accel_buffer, -num_new, axis=0)
		self.gyro_buffer = np.roll(self.gyro_buffer, -num_new, axis=0)

		for i, sample in enumerate(imu_samples):
			self.accel_buffer[-num_new + i, :] = sample.acc
			self.gyro_buffer[-num_new + i, :] = sample.gyro

		for i in range(3):
			self.lines_a[i].set_data(
				pos=np.column_stack((self.x_axis, self.accel_buffer[:, i])))
			self.lines_g[i].set_data(
				pos=np.column_stack((self.x_axis, self.gyro_buffer[:, i])))

		self.canvas.app.process_events()

	def _update_fps(self):
		self._frames += 1
		now = perf_counter()
		dt = now - self._last_time
		if dt >= 1.0:
			fps = self._frames / dt
			self.fps_text.text = f"FPS: {fps:.1f}"
			self.fps_text.pos = (10, self.canvas.size[1] - 10)
			self._frames = 0
			self._last_time = now
