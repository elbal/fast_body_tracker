import numpy as np
from numpy import typing as npt

from pykinect_azure.k4abt._k4abtTypes import k4abt_body_t
from pykinect_azure.k4abt.kabt_const import K4ABT_JOINT_COUNT

JOINT_DTYPE = np.dtype([
	("position", np.float32, 3),
	("orientation", np.float32, 4),
	("confidence", np.int32)
	])


class Body:
	def __init__(self, body_handle: k4abt_body_t):
		self._handle = body_handle
		self.joints_data = np.frombuffer(
			self._handle.skeleton.joints, dtype=JOINT_DTYPE,
			count=K4ABT_JOINT_COUNT)

	def handle(self):
		return self._handle

	@property
	def positions(self) -> npt.NDArray[np.float32]:
		return self.joints_data["position"]

	@property
	def orientations(self) -> npt.NDArray[np.float32]:
		return self.joints_data["orientation"]

	@property
	def confidences(self) -> npt.NDArray[np.int32]:
		return self.joints_data['confidence']
