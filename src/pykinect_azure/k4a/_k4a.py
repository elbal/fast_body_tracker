from ._k4a_types import *


class AzureKinectSensorException(Exception):
	pass


class K4aLib:
	_dll = None

	k4a_device_get_installed_count = None
	k4a_device_open = None
	k4a_device_close = None
	k4a_device_get_capture = None
	k4a_device_get_imu_sample = None
	k4a_capture_create = None
	k4a_capture_release = None
	k4a_capture_reference = None
	k4a_capture_get_color_image = None
	k4a_capture_get_depth_image = None
	k4a_capture_get_ir_image = None
	k4a_capture_set_color_image = None
	k4a_capture_set_depth_image = None
	k4a_capture_set_ir_image = None
	k4a_capture_set_temperature_c = None
	k4a_capture_get_temperature_c = None
	k4a_image_create = None
	k4a_image_create_from_buffer = None
	k4a_image_get_buffer = None
	k4a_image_get_size = None
	k4a_image_get_format = None
	k4a_image_get_width_pixels = None
	k4a_image_get_height_pixels = None
	k4a_image_get_stride_bytes = None
	k4a_image_get_device_timestamp_usec = None
	k4a_image_get_system_timestamp_nsec = None
	k4a_image_get_exposure_usec = None
	k4a_image_get_white_balance = None
	k4a_image_get_iso_speed = None
	k4a_image_set_device_timestamp_usec = None
	k4a_image_set_system_timestamp_nsec = None
	k4a_image_set_exposure_usec = None
	k4a_image_set_white_balance = None
	k4a_image_set_iso_speed = None
	k4a_image_reference = None
	k4a_image_release = None
	k4a_device_start_cameras = None
	k4a_device_stop_cameras = None
	k4a_device_start_imu = None
	k4a_device_stop_imu = None
	k4a_device_get_serialnum = None
	k4a_device_get_version = None
	k4a_device_get_color_control_capabilities = None
	k4a_device_get_color_control = None
	k4a_device_set_color_control = None
	k4a_device_get_raw_calibration = None
	k4a_device_get_calibration = None
	k4a_device_get_sync_jack = None
	k4a_calibration_get_from_raw = None
	k4a_calibration_3d_to_3d = None
	k4a_calibration_2d_to_3d = None
	k4a_calibration_3d_to_2d = None
	k4a_calibration_2d_to_2d = None
	k4a_calibration_color_2d_to_depth_2d = None
	k4a_transformation_create = None
	k4a_transformation_destroy = None
	k4a_transformation_depth_image_to_color_camera = None
	k4a_transformation_depth_image_to_color_camera_custom = None
	k4a_transformation_color_image_to_depth_camera = None
	k4a_transformation_depth_image_to_point_cloud = None

	@classmethod
	def setup(cls, path):
		cls._dll = ctypes.CDLL(path)
		cls._bind_all()

	@classmethod
	def _bind(cls, name, restype, argtypes):
		func = getattr(cls._dll, name)
		func.restype = restype
		func.argtypes = argtypes
		setattr(cls, name, func)

	@classmethod
	def _bind_all(cls):
		cls._bind("k4a_device_get_installed_count", ctypes.c_uint32, [])
		cls._bind("k4a_device_open", ctypes.c_int, (ctypes.c_uint32, ctypes.POINTER(k4a_device_t)))
		cls._bind("k4a_device_close", None, (k4a_device_t,))
		cls._bind("k4a_device_get_capture", ctypes.c_int, (k4a_device_t, ctypes.POINTER(k4a_capture_t), ctypes.c_int32))
		cls._bind("k4a_device_get_imu_sample", ctypes.c_int, (k4a_device_t, ctypes.POINTER(k4a_imu_sample_t), ctypes.c_int32))
		cls._bind("k4a_device_start_cameras", k4a_result_t, (k4a_device_t, ctypes.POINTER(k4a_device_configuration_t)))
		cls._bind("k4a_device_stop_cameras", None, (k4a_device_t,))
		cls._bind("k4a_device_start_imu", k4a_result_t, (k4a_device_t,))
		cls._bind("k4a_device_stop_imu", None, (k4a_device_t,))
		cls._bind("k4a_device_get_serialnum", k4a_buffer_result_t, (k4a_device_t, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)))
		cls._bind("k4a_device_get_version", k4a_result_t, (k4a_device_t, ctypes.POINTER(k4a_hardware_version_t)))
		cls._bind("k4a_device_get_color_control_capabilities", k4a_result_t, (k4a_device_t, k4a_color_control_command_t, ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(k4a_color_control_mode_t)))
		cls._bind("k4a_device_get_color_control", k4a_result_t, (k4a_device_t, k4a_color_control_command_t, ctypes.POINTER(k4a_color_control_mode_t), ctypes.POINTER(ctypes.c_int32)))
		cls._bind("k4a_device_set_color_control", k4a_result_t, (k4a_device_t, k4a_color_control_command_t, k4a_color_control_mode_t, ctypes.c_int32))
		cls._bind("k4a_device_get_raw_calibration", k4a_buffer_result_t, (k4a_device_t, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_size_t)))
		cls._bind("k4a_device_get_calibration", k4a_result_t, (k4a_device_t, k4a_depth_mode_t, k4a_color_resolution_t, ctypes.POINTER(k4a_calibration_t)))
		cls._bind("k4a_device_get_sync_jack", k4a_result_t, (k4a_device_t, ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool)))

		cls._bind("k4a_capture_create", k4a_result_t, (ctypes.POINTER(k4a_capture_t),))
		cls._bind("k4a_capture_release", None, (k4a_capture_t,))
		cls._bind("k4a_capture_reference", None, (k4a_capture_t,))
		cls._bind("k4a_capture_get_color_image", k4a_image_t, (k4a_capture_t,))
		cls._bind("k4a_capture_get_depth_image", k4a_image_t, (k4a_capture_t,))
		cls._bind("k4a_capture_get_ir_image", k4a_image_t, (k4a_capture_t,))
		cls._bind("k4a_capture_set_color_image", None, (k4a_capture_t, k4a_image_t))
		cls._bind("k4a_capture_set_depth_image", None, (k4a_capture_t, k4a_image_t))
		cls._bind("k4a_capture_set_ir_image", None, (k4a_capture_t, k4a_image_t))
		cls._bind("k4a_capture_set_temperature_c", None, (k4a_capture_t, ctypes.c_float))
		cls._bind("k4a_capture_get_temperature_c", ctypes.c_float, (k4a_capture_t,))

		cls._bind("k4a_image_create", k4a_result_t, (k4a_image_format_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(k4a_image_t)))
		cls._bind("k4a_image_create_from_buffer", k4a_result_t, (k4a_image_format_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(k4a_image_t)))
		cls._bind("k4a_image_get_buffer", ctypes.POINTER(ctypes.c_uint8), (k4a_image_t,))
		cls._bind("k4a_image_get_size", ctypes.c_size_t, (k4a_image_t,))
		cls._bind("k4a_image_get_format", k4a_image_format_t, (k4a_image_t,))
		cls._bind("k4a_image_get_width_pixels", ctypes.c_int, (k4a_image_t,))
		cls._bind("k4a_image_get_height_pixels", ctypes.c_int, (k4a_image_t,))
		cls._bind("k4a_image_get_stride_bytes", ctypes.c_int, (k4a_image_t,))
		cls._bind("k4a_image_get_device_timestamp_usec", ctypes.c_uint64, (k4a_image_t,))
		cls._bind("k4a_image_get_system_timestamp_nsec", ctypes.c_uint64, (k4a_image_t,))
		cls._bind("k4a_image_get_exposure_usec", ctypes.c_uint64, (k4a_image_t,))
		cls._bind("k4a_image_get_white_balance", ctypes.c_uint32, (k4a_image_t,))
		cls._bind("k4a_image_get_iso_speed", ctypes.c_uint32, (k4a_image_t,))
		cls._bind("k4a_image_set_device_timestamp_usec", None, (k4a_image_t, ctypes.c_uint64))
		cls._bind("k4a_image_set_system_timestamp_nsec", None, (k4a_image_t, ctypes.c_uint64))
		cls._bind("k4a_image_set_exposure_usec", None, (k4a_image_t, ctypes.c_uint64))
		cls._bind("k4a_image_set_white_balance", None, (k4a_image_t, ctypes.c_uint32))
		cls._bind("k4a_image_set_iso_speed", None, (k4a_image_t, ctypes.c_uint32))
		cls._bind("k4a_image_reference", None, (k4a_image_t,))
		cls._bind("k4a_image_release", None, (k4a_image_t,))

		cls._bind("k4a_calibration_get_from_raw", k4a_result_t, (ctypes.POINTER(ctypes.c_char), ctypes.c_size_t, k4a_depth_mode_t, k4a_color_resolution_t, ctypes.POINTER(k4a_calibration_t)))
		cls._bind("k4a_calibration_3d_to_3d", k4a_result_t, (ctypes.POINTER(k4a_calibration_t), ctypes.POINTER(k4a_float3), k4a_calibration_type_t, k4a_calibration_type_t, ctypes.POINTER(k4a_float3)))
		cls._bind("k4a_calibration_2d_to_3d", k4a_result_t, (ctypes.POINTER(k4a_calibration_t), ctypes.POINTER(k4a_float2_t), ctypes.c_float, k4a_calibration_type_t, k4a_calibration_type_t, ctypes.POINTER(k4a_float3), ctypes.POINTER(ctypes.c_int)))
		cls._bind("k4a_calibration_3d_to_2d", k4a_result_t, (ctypes.POINTER(k4a_calibration_t), ctypes.POINTER(k4a_float3), k4a_calibration_type_t, k4a_calibration_type_t, ctypes.POINTER(k4a_float2_t), ctypes.POINTER(ctypes.c_int)))
		cls._bind("k4a_calibration_2d_to_2d", k4a_result_t, (ctypes.POINTER(k4a_calibration_t), ctypes.POINTER(k4a_float2_t), ctypes.c_float, k4a_calibration_type_t, k4a_calibration_type_t, ctypes.POINTER(k4a_float2_t), ctypes.POINTER(ctypes.c_int)))
		cls._bind("k4a_calibration_color_2d_to_depth_2d", k4a_result_t, (ctypes.POINTER(k4a_calibration_t), ctypes.POINTER(k4a_float2_t), k4a_image_t, ctypes.POINTER(k4a_float2_t), ctypes.POINTER(ctypes.c_int)))

		cls._bind("k4a_transformation_create", k4a_transformation_t, (ctypes.POINTER(k4a_calibration_t),))
		cls._bind("k4a_transformation_destroy", None, (k4a_transformation_t,))
		cls._bind("k4a_transformation_depth_image_to_color_camera", k4a_result_t, (k4a_transformation_t, k4a_image_t, k4a_image_t))
		cls._bind("k4a_transformation_depth_image_to_color_camera_custom", k4a_result_t, (k4a_transformation_t, k4a_image_t, k4a_image_t, k4a_image_t, k4a_image_t, k4a_transformation_interpolation_type_t, ctypes.c_uint32))
		cls._bind("k4a_transformation_color_image_to_depth_camera", k4a_result_t, (k4a_transformation_t, k4a_image_t, k4a_image_t, k4a_image_t))
		cls._bind("k4a_transformation_depth_image_to_point_cloud", k4a_result_t, (k4a_transformation_t, k4a_image_t, k4a_calibration_type_t, k4a_image_t))


def setup_library(path):
	K4aLib.setup(path)


def k4a_device_get_installed_count():
	return K4aLib.k4a_device_get_installed_count()


def k4a_device_open(id, h):
	return K4aLib.k4a_device_open(id, h)


def k4a_device_close(h):
	K4aLib.k4a_device_close(h)


def k4a_device_get_capture(h, ch, t):
	return K4aLib.k4a_device_get_capture(h, ch, t)


def k4a_device_get_imu_sample(h, ih, t):
	return K4aLib.k4a_device_get_imu_sample(h, ih, t)


def k4a_capture_create(ch):
	return K4aLib.k4a_capture_create(ch)


def k4a_capture_release(ch):
	K4aLib.k4a_capture_release(ch)


def k4a_capture_reference(ch):
	K4aLib.k4a_capture_reference(ch)


def k4a_capture_get_color_image(ch):
	return K4aLib.k4a_capture_get_color_image(ch)

def k4a_capture_get_depth_image(ch):
	return K4aLib.k4a_capture_get_depth_image(ch)

def k4a_capture_get_ir_image(ch):
	return K4aLib.k4a_capture_get_ir_image(ch)

def k4a_capture_set_color_image(ch, ih):
	K4aLib.k4a_capture_set_color_image(ch, ih)

def k4a_capture_set_depth_image(ch, ih):
	K4aLib.k4a_capture_set_depth_image(ch, ih)

def k4a_capture_set_ir_image(ch, ih): K4aLib.k4a_capture_set_ir_image(ch, ih)
def k4a_capture_set_temperature_c(ch, t): K4aLib.k4a_capture_set_temperature_c(ch, t)
def k4a_capture_get_temperature_c(ch): return K4aLib.k4a_capture_get_temperature_c(ch)
def k4a_image_create(f, w, h, s, ih): return K4aLib.k4a_image_create(f, w, h, s, ih)
def k4a_image_create_from_buffer(f, w, h, s, b, bs, cb, ctx, ih): return K4aLib.k4a_image_create_from_buffer(f, w, h, s, b, bs, cb, ctx, ih)
def k4a_image_get_buffer(ih): return K4aLib.k4a_image_get_buffer(ih)
def k4a_image_get_size(ih): return K4aLib.k4a_image_get_size(ih)
def k4a_image_get_format(ih): return K4aLib.k4a_image_get_format(ih)
def k4a_image_get_width_pixels(ih): return K4aLib.k4a_image_get_width_pixels(ih)
def k4a_image_get_height_pixels(ih): return K4aLib.k4a_image_get_height_pixels(ih)
def k4a_image_get_stride_bytes(ih): return K4aLib.k4a_image_get_stride_bytes(ih)
def k4a_image_get_device_timestamp_usec(ih): return K4aLib.k4a_image_get_device_timestamp_usec(ih)
def k4a_image_get_system_timestamp_nsec(ih): return K4aLib.k4a_image_get_system_timestamp_nsec(ih)
def k4a_image_get_exposure_usec(ih): return K4aLib.k4a_image_get_exposure_usec(ih)
def k4a_image_get_white_balance(ih): return K4aLib.k4a_image_get_white_balance(ih)
def k4a_image_get_iso_speed(ih): return K4aLib.k4a_image_get_iso_speed(ih)
def k4a_image_set_device_timestamp_usec(ih, t): K4aLib.k4a_image_set_device_timestamp_usec(ih, t)
def k4a_image_set_system_timestamp_nsec(ih, t): K4aLib.k4a_image_set_system_timestamp_nsec(ih, t)
def k4a_image_set_exposure_usec(ih, e): K4aLib.k4a_image_set_exposure_usec(ih, e)
def k4a_image_set_white_balance(ih, w): K4aLib.k4a_image_set_white_balance(ih, w)
def k4a_image_set_iso_speed(ih, i): K4aLib.k4a_image_set_iso_speed(ih, i)
def k4a_image_reference(ih): K4aLib.k4a_image_reference(ih)
def k4a_image_release(ih): K4aLib.k4a_image_release(ih)
def k4a_device_start_cameras(h, c): return K4aLib.k4a_device_start_cameras(h, c)
def k4a_device_stop_cameras(h): K4aLib.k4a_device_stop_cameras(h)
def k4a_device_start_imu(h): return K4aLib.k4a_device_start_imu(h)
def k4a_device_stop_imu(h): K4aLib.k4a_device_stop_imu(h)
def k4a_device_get_serialnum(h, sn, sns): return K4aLib.k4a_device_get_serialnum(h, sn, sns)
def k4a_device_get_version(h, v): return K4aLib.k4a_device_get_version(h, v)
def k4a_device_get_color_control_capabilities(h, c, sa, minv, maxv, sv, dv, dm): return K4aLib.k4a_device_get_color_control_capabilities(h, c, sa, minv, maxv, sv, dv, dm)
def k4a_device_get_color_control(h, c, m, v): return K4aLib.k4a_device_get_color_control(h, c, m, v)
def k4a_device_set_color_control(h, c, m, v): return K4aLib.k4a_device_set_color_control(h, c, m, v)
def k4a_device_get_raw_calibration(h, d, ds): return K4aLib.k4a_device_get_raw_calibration(h, d, ds)
def k4a_device_get_calibration(h, dm, cr, c): return K4aLib.k4a_device_get_calibration(h, dm, cr, c)
def k4a_device_get_sync_jack(h, sij, soj): return K4aLib.k4a_device_get_sync_jack(h, sij, soj)
def k4a_calibration_get_from_raw(rc, rcs, dm, cr, c): return K4aLib.k4a_calibration_get_from_raw(rc, rcs, dm, cr, c)
def k4a_calibration_3d_to_3d(c, s3, sc, tc, t3): return K4aLib.k4a_calibration_3d_to_3d(c, s3, sc, tc, t3)
def k4a_calibration_2d_to_3d(c, s2, sd, sc, tc, t3, v): return K4aLib.k4a_calibration_2d_to_3d(c, s2, sd, sc, tc, t3, v)
def k4a_calibration_3d_to_2d(c, s3, sc, tc, t2, v): return K4aLib.k4a_calibration_3d_to_2d(c, s3, sc, tc, t2, v)
def k4a_calibration_2d_to_2d(c, s2, sd, sc, tc, t2, v): return K4aLib.k4a_calibration_2d_to_2d(c, s2, sd, sc, tc, t2, v)
def k4a_calibration_color_2d_to_depth_2d(c, s2, di, t2, v): return K4aLib.k4a_calibration_color_2d_to_depth_2d(c, s2, di, t2, v)
def k4a_transformation_create(c): return K4aLib.k4a_transformation_create(c)
def k4a_transformation_destroy(h): K4aLib.k4a_transformation_destroy(h)
def k4a_transformation_depth_image_to_color_camera(h, di, tdi): return K4aLib.k4a_transformation_depth_image_to_color_camera(h, di, tdi)
def k4a_transformation_depth_image_to_color_camera_custom(h, di, ci, tdi, tci, it, icv): return K4aLib.k4a_transformation_depth_image_to_color_camera_custom(h, di, ci, tdi, tci, it, icv)
def k4a_transformation_color_image_to_depth_camera(h, di, ci, tci): return K4aLib.k4a_transformation_color_image_to_depth_camera(h, di, ci, tci)
def k4a_transformation_depth_image_to_point_cloud(h, di, c, xyzi): return K4aLib.k4a_transformation_depth_image_to_point_cloud(h, di, c, xyzi)


def verify(result, error):
	if result != K4A_RESULT_SUCCEEDED:
		raise AzureKinectSensorException(error)
