import numpy as np
from numpy import typing as npt
import cv2

from pykinect_azure.k4a._k4a_types import K4A_CALIBRATION_TYPE_DEPTH
from pykinect_azure.k4abt._k4abt_types import (
	body_colors, k4abt_body_t, K4ABT_JOINT_COUNT, K4ABT_SEGMENT_PAIRS)
from pykinect_azure.k4a import Calibration, Image

JOINT2D_DTYPE = np.dtype([
	("position", np.float32, 2), ("confidence", np.int32, 1)])


class Body2d:
	def __init__(
			self, body_handle: k4abt_body_t, calibration: Calibration,
			target_camera: int = K4A_CALIBRATION_TYPE_DEPTH):
		self.id = body_handle.id
		self.joints_data = np.zeros(K4ABT_JOINT_COUNT, dtype=JOINT2D_DTYPE)
		for i in range(K4ABT_JOINT_COUNT):
			joint = body_handle.skeleton.joints[i]
			position_2d_handle = calibration.convert_3d_to_2d(
				source_point3d=joint.position,
				source_camera=K4A_CALIBRATION_TYPE_DEPTH,
				target_camera=target_camera)
			self.joints_data["position"][i] = position_2d_handle.v[:]
			self.joints_data["confidence"][i] = joint.confidence_level

	@property
	def positions(self) -> npt.NDArray[np.float32]:
		return self.joints_data["position"]

	@property
	def confidences(self) -> npt.NDArray[np.int32]:
		return self.joints_data["confidence"]

	def draw(self, image: Image, only_segments=False) -> Image:
		positions = self.positions.astype(np.int32)
		confidences = self.confidences
		color = (
			int(body_colors[self.id][0]),
			int(body_colors[self.id][1]),
			int(body_colors[self.id][2]))

		for segment_pair in K4ABT_SEGMENT_PAIRS:
			idx1, idx2 = segment_pair
			point1 = tuple(positions[idx1])
			point2 = tuple(positions[idx2])
			if (
					(point1[0] == 0 and point1[1] == 0)
					or (point2[0] == 0 and point2[1] == 0)
					or confidences[idx1] == 0
					or confidences[idx2] == 0):
				continue
			image = cv2.line(image, point1, point2, color, 2)
		if only_segments:
			return image

		for i in range(len(positions)):
			point = tuple(positions[i])
			if (point[0] == 0 and point[1] == 0) or confidences[i] == 0:
				continue
			image = cv2.circle(image, point, 3, color, 3)

		return image
