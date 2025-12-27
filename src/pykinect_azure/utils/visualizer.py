import numpy as np
from numpy import typing as npt
from time import perf_counter
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Markers, Text
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
