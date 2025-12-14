import ctypes

from pykinect_azure.k4abt.kabt_const import *
from pykinect_azure.k4a._k4atypes import k4a_float2_t, k4a_float3

k4abt_result_t = ctypes.c_int
k4abt_float4 = ctypes.c_float * 4


# K4A_DECLARE_HANDLE(k4abt_tracker_t);
class _handle_k4abt_tracker_t(ctypes.Structure):
	_fields_ = [("_rsvd", ctypes.c_size_t)]


k4abt_tracker_t = ctypes.POINTER(_handle_k4abt_tracker_t)


# K4A_DECLARE_HANDLE(k4abt_frame_t);
class _handle_k4abt_frame_t(ctypes.Structure):
	_fields_ = [("_rsvd", ctypes.c_size_t)]


k4abt_frame_t = ctypes.POINTER(_handle_k4abt_frame_t)


class _k4abt_tracker_configuration_t(ctypes.Structure):
	_fields_ = [
		("sensor_orientation", ctypes.c_int),
		("processing_mode", ctypes.c_int),
		("gpu_device_id", ctypes.c_int32),
		("model_path", ctypes.c_char_p)
		]


k4abt_tracker_configuration_t = _k4abt_tracker_configuration_t
k4abt_tracker_default_configuration = k4abt_tracker_configuration_t()
k4abt_tracker_default_configuration.sensor_orientation = (
	K4ABT_SENSOR_ORIENTATION_DEFAULT)
k4abt_tracker_default_configuration.processing_mode = (
	K4ABT_TRACKER_PROCESSING_MODE_GPU)
k4abt_tracker_default_configuration.gpu_device_id = 0


class _k4abt_joint_t(ctypes.Structure):
	_fields_ = [
		("position", k4a_float3), ("orientation", k4abt_float4),
		("confidence_level", ctypes.c_int)]

k4abt_joint_t = _k4abt_joint_t


class k4abt_skeleton_t(ctypes.Structure):
	_fields_ = [("joints", _k4abt_joint_t * K4ABT_JOINT_COUNT)]


class k4abt_body_t(ctypes.Structure):
	_fields_ = [("id", ctypes.c_uint32), ("skeleton", k4abt_skeleton_t)]


class _k4abt_joint2D_t(ctypes.Structure):
	_fields_ = [("position", k4a_float2_t), ("confidence_level", ctypes.c_int)]

	def __init__(self, position=(0, 0), confidence_level=0):
		super().__init__()
		self.position = k4a_float2_t(position)
		self.confidence_level = confidence_level

	def __iter__(self):
		return {
			"position": self.position.__iter__(),
			"confidence_level": self.confidence_level}


k4abt_joint2D_t = _k4abt_joint2D_t


class k4abt_skeleton2D_t(ctypes.Structure):
	_fields_ = [("joints2D", _k4abt_joint2D_t * K4ABT_JOINT_COUNT)]

	def __init__(
			self,
			joints=(_k4abt_joint2D_t() for i in range(K4ABT_JOINT_COUNT))):
		super().__init__()
		self.joints2D = (_k4abt_joint2D_t * K4ABT_JOINT_COUNT)(*joints)

	def __iter__(self):
		return {"joints2D": [joint.__iter__() for joint in self.joints2D]}


class k4abt_body2D_t(ctypes.Structure):
	_fields_ = [("id", ctypes.c_uint32), ("skeleton", k4abt_skeleton2D_t)]

	def __init__(self, id=0, skeleton=k4abt_skeleton2D_t()):
		super().__init__()
		self.id = id
		self.skeleton = skeleton

	def __iter__(self):
		return {"id": self.id, "skeleton": self.skeleton.__iter__()}
