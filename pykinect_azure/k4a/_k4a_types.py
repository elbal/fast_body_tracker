import ctypes

from pykinect_azure.k4a.k4a_const import *

k4a_result_t = ctypes.c_int
k4a_buffer_result_t = ctypes.c_int
k4a_wait_result_t = ctypes.c_int
k4a_log_level_t = ctypes.c_int
k4a_depth_mode_t = ctypes.c_int
k4a_color_resolution_t = ctypes.c_int
k4a_image_format_t = ctypes.c_int
k4a_transformation_interpolation_type_t = ctypes.c_int
k4a_fps_t = ctypes.c_int
k4a_color_control_command_t = ctypes.c_int
k4a_color_control_mode_t = ctypes.c_int
k4a_wired_sync_mode_t = ctypes.c_int
k4a_calibration_type_t = ctypes.c_int
k4a_calibration_model_type_t = ctypes.c_int
k4a_firmware_build_t = ctypes.c_int
k4a_firmware_signature_t = ctypes.c_int
k4a_float3 = ctypes.c_float * 3
k4a_float2 = ctypes.c_float * 2


class _handle_k4a_device_t(ctypes.Structure):
	_fields_ = [("_rsvd", ctypes.c_size_t),]


k4a_device_t = ctypes.POINTER(_handle_k4a_device_t)


class _handle_k4a_capture_t(ctypes.Structure):
	_fields_ = [("_rsvd", ctypes.c_size_t),]


k4a_capture_t = ctypes.POINTER(_handle_k4a_capture_t)


class _handle_k4a_image_t(ctypes.Structure):
	_fields_ = [("_rsvd", ctypes.c_size_t),]


k4a_image_t = ctypes.POINTER(_handle_k4a_image_t)


class _handle_k4a_transformation_t(ctypes.Structure):
	_fields_ = [("_rsvd", ctypes.c_size_t),]


k4a_transformation_t = ctypes.POINTER(_handle_k4a_transformation_t)


def K4A_SUCCEEDED(result):
	return result == K4A_RESULT_SUCCEEDED


def K4A_FAILED(result):
	return not K4A_SUCCEEDED(result)


class _k4a_device_configuration_t(ctypes.Structure):
	_fields_ = [
		("color_format", ctypes.c_int), ("color_resolution", ctypes.c_int),
		("depth_mode", ctypes.c_int), ("camera_fps", ctypes.c_int),
		("synchronized_images_only", ctypes.c_bool),
		("depth_delay_off_color_usec", ctypes.c_int32),
		("wired_sync_mode", ctypes.c_int),
		("subordinate_delay_off_master_usec", ctypes.c_uint32),
		("disable_streaming_indicator", ctypes.c_bool),
		]


k4a_device_configuration_t = _k4a_device_configuration_t


class _k4a_calibration_extrinsics_t(ctypes.Structure):
	_fields_ = [
		("rotation", ctypes.c_float * 9), ("translation", ctypes.c_float * 3),]


k4a_calibration_extrinsics_t = _k4a_calibration_extrinsics_t


class _param(ctypes.Structure):
	_fields_ = [
		("cx", ctypes.c_float), ("cy", ctypes.c_float), ("fx", ctypes.c_float),
		("fy", ctypes.c_float), ("k1", ctypes.c_float), ("k2", ctypes.c_float),
		("k3", ctypes.c_float), ("k4", ctypes.c_float), ("k5", ctypes.c_float),
		("k6", ctypes.c_float), ("codx", ctypes.c_float),
		("cody", ctypes.c_float), ("p2", ctypes.c_float),
		("p1", ctypes.c_float), ("metric_radius", ctypes.c_float),
		]


class k4a_calibration_intrinsic_parameters_t(ctypes.Union):
	_fields_ = [("param", _param), ("v", ctypes.c_float * 15),]


class _k4a_calibration_intrinsics_t(ctypes.Structure):
	_fields_ = [
		("type", ctypes.c_int), ("parameter_count", ctypes.c_uint),
		("parameters", k4a_calibration_intrinsic_parameters_t),
		]


k4a_calibration_intrinsics_t = _k4a_calibration_intrinsics_t

class _k4a_calibration_camera_t(ctypes.Structure):
	_fields_ = [
		("extrinsics", k4a_calibration_extrinsics_t),
		("intrinsics", k4a_calibration_intrinsics_t),
		("resolution_width", ctypes.c_int),
		("resolution_height", ctypes.c_int),
		("metric_radius", ctypes.c_float),
		]


k4a_calibration_camera_t = _k4a_calibration_camera_t


class _k4a_calibration_t(ctypes.Structure):
	_fields_ = [
		("depth_camera_calibration", k4a_calibration_camera_t),
		("color_camera_calibration", k4a_calibration_camera_t),
		("extrinsics", (
				k4a_calibration_extrinsics_t * K4A_CALIBRATION_TYPE_NUM
				* K4A_CALIBRATION_TYPE_NUM)),
		("depth_mode", ctypes.c_int), ("color_resolution", ctypes.c_int),
		]


k4a_calibration_t = _k4a_calibration_t


class _k4a_version_t(ctypes.Structure):
	_fields_ = [
		("major", ctypes.c_uint32), ("minor", ctypes.c_uint32),
		("iteration", ctypes.c_uint32),
		]


k4a_version_t = _k4a_version_t


class _k4a_hardware_version_t(ctypes.Structure):
	_fields_ = [
		("rgb", k4a_version_t), ("depth", k4a_version_t),
		("audio", k4a_version_t), ("depth_sensor", k4a_version_t),
		("firmware_build", ctypes.c_int), ("firmware_signature", ctypes.c_int),
		]


k4a_hardware_version_t = _k4a_hardware_version_t


class _xy(ctypes.Structure):
	_fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float),]

	def __iter__(self):
		return {'x':self.x, 'y':self.y}

	def __str__(self):
		return str(self.__iter__())


class k4a_float2_t(ctypes.Union):
	_fields_ = [("xy", _xy), ("v", ctypes.c_float * 2)]

	def __init__(self, v=(0,0)):
		super().__init__()
		self.xy = _xy(v[0], v[1])

	def __iter__(self):
		xy = self.xy.__iter__()
		xy.update({'v':[v for v in self.v]})
		return xy

	def __str__(self):
		return self.xy.__str__()


class _xyz(ctypes.Structure):
	_fields_ = [
		("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float),]

	def __iter__(self):
		return {'x':self.x, 'y':self.y, 'z':self.z}

	def __str__(self):
		return str(self.__iter__())


class k4a_float3_t(ctypes.Union):
	_fields_ = [("xyz", _xyz), ("v", ctypes.c_float * 3)]

	def __init__(self, v=(0,0,0)):
		super().__init__()
		self.xyz = _xyz(v[0], v[1], v[2])

	def __iter__(self):
		xyz = self.xyz.__iter__()
		xyz.update({'v':[v for v in self.v]})
		return xyz

	def __str__(self):
		return self.xyz.__str__()


class k4a_imu_sample_t(ctypes.Structure):
	_fields_ = [
		("temperature", ctypes.c_float),
		("acc_sample", k4a_float3_t),
		("acc_timestamp_usec", ctypes.c_uint64),
		("gyro_sample", k4a_float3_t),
		("gyro_timestamp_usec", ctypes.c_uint64),
	]


IMU_SAMPLE_SIZE = ctypes.sizeof(k4a_imu_sample_t)
