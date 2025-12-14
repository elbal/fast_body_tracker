import numpy as np
from pykinect_azure.k4abt._k4abtTypes import k4abt_skeleton_t, K4ABT_JOINT_NAMES


class Joint:
	def __init__(self, skeleton_handle: k4abt_skeleton_t, joint_id: int):
		self._handle = skeleton_handle
		self.position = skeleton_handle.position.xyz
		self.orientation = skeleton_handle.orientation.wxyz
		self.confidence_level = skeleton_handle.confidence_level
		self.id = joint_id
		self.name = self.get_name()

	def numpy(self):
		return np.array([self.position.x, self.position.y, self.position.z,
						 self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z,
						 self.confidence_level])

	def is_valid(self):
		return self._handle

	def handle(self):
		return self._handle

	def get_name(self):
		return K4ABT_JOINT_NAMES[self.id]
