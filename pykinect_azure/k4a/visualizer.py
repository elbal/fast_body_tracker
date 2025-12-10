import numpy as np
import cv2
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Markers
from vispy.scene.cameras import TurntableCamera


class PointCloudVisualizer:
	def __init__(self):
		self.canvas = SceneCanvas(
			keys="interactive", show=True, title="Point cloud")
		self.view = self.canvas.central_widget.add_view()
		self.view.camera = TurntableCamera(
			fov=45, distance=3.0, up="-y", elevation=30, azimuth=0,
			flip=(False, True, False))
		self.view.camera.set_range(x=[-2, 2], y=[-2, 2], z=[-2, 2])

		self.flip_vector = np.array([1, -1, -1], dtype=np.float32)
		self.scatter = Markers(parent=self.view.scene)

		self.first_valid_frame = True
		self.initial_center = np.array(
			[0, 0, 1.5], dtype=np.float32)
		self.points_buffer = np.zeros(
			(1000, 3), dtype=np.float32)

	def __call__(self, points_3d, rgb_image=None):
		self.update(points_3d, rgb_image)

	def update(self, points_3d, rgb_image=None):
		if points_3d is None or len(points_3d) == 0:
			return
		if rgb_image is not None:
			colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB).reshape(-1, 3)
			colors = colors.astype(np.float32) / 255.0
		else:
			colors = (1, 1, 1)

		points_3d = points_3d * self.flip_vector
		valid_mask = ~np.all(points_3d == 0, axis=1)
		points_filtered = points_3d[valid_mask]
		colors_filtered = colors[valid_mask]
		if points_filtered.size == 0:
			return

		if self.first_valid_frame:
			min_xyz = points_filtered.min(axis=0)
			max_xyz = points_filtered.max(axis=0)

			self.initial_center = (max_xyz + min_xyz) / 2
			self.view.camera.center = tuple(self.initial_center)

			radius = np.linalg.norm(max_xyz - min_xyz) / 2
			if radius > 0.5:
				self.view.camera.distance = radius * 2.5

			self.first_valid_frame = False

		self.scatter.set_data(
			pos=points_filtered, face_color=colors_filtered, size=2,
			edge_width=0)
		self.canvas.app.process_events()
