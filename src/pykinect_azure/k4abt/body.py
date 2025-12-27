import numpy as np
from numpy import typing as npt

from pykinect_azure.k4abt._k4abt_types import k4abt_body_t
from pykinect_azure.k4abt.kabt_const import K4ABT_JOINT_COUNT

JOINT_DTYPE = np.dtype([
	("position", np.float32, 3),
	("orientation", np.float32, 4),
	("confidence", np.int32)
	])


class Body:
	def __init__(self, body_handle: k4abt_body_t):
		self._handle = body_handle
		joints = np.ctypeslib.as_array(
			self._handle.skeleton.joints, shape=(K4ABT_JOINT_COUNT,))
		self.joints_data = joints.view(JOINT_DTYPE)

	@property
	def positions(self) -> npt.NDArray[np.float32]:
		return self.joints_data["position"]

	@property
	def orientations(self) -> npt.NDArray[np.float32]:
		return self.joints_data["orientation"]

	@property
	def confidences(self) -> npt.NDArray[np.int32]:
		return self.joints_data['confidence']
