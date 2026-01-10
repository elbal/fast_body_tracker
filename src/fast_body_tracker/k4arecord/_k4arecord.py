import sys
import traceback

from ._k4arecordTypes import *
from ..k4a._k4a_types import *

record_dll = None


def setup_library(module_k4arecord_path):
    global record_dll

    try:
        record_dll = ctypes.CDLL(module_k4arecord_path)
    except Exception as e:
        print("Failed to load library", e)
        sys.exit(1)


def k4a_record_create(file_path, device, device_config, recording_handle):
    _k4a_record_create = record_dll.k4a_record_create
    _k4a_record_create.restype = k4a_result_t
    _k4a_record_create.argtypes = (
        ctypes.POINTER(ctypes.c_char), k4a_device_t,
        k4a_device_configuration_t, ctypes.POINTER(k4a_record_t),)

    return _k4a_record_create(
        file_path, device, device_config, recording_handle)


def k4a_record_write_header(recording_handle):
    _k4a_record_write_header = record_dll.k4a_record_write_header
    _k4a_record_write_header.restype = k4a_result_t
    _k4a_record_write_header.argtypes = (k4a_record_t,)

    return _k4a_record_write_header(recording_handle)


def k4a_record_write_capture(recording_handle, capture_handle):
    _k4a_record_write_capture = record_dll.k4a_record_write_capture
    _k4a_record_write_capture.restype = k4a_result_t
    _k4a_record_write_capture.argtypes = (k4a_record_t, k4a_capture_t,)

    return _k4a_record_write_capture(recording_handle, capture_handle)


def k4a_record_flush(recording_handle):
    _k4a_record_flush = record_dll.k4a_record_flush
    _k4a_record_flush.restype = k4a_result_t
    _k4a_record_flush.argtypes = (k4a_record_t,)

    return _k4a_record_flush(recording_handle)


def k4a_record_close(recording_handle):
    _k4a_record_close = record_dll.k4a_record_close
    _k4a_record_close.restype = None
    _k4a_record_close.argtypes = (k4a_record_t,)

    _k4a_record_close(recording_handle)


def k4a_playback_open(file_path, playback_handle):
    _k4a_playback_open = record_dll.k4a_playback_open
    _k4a_playback_open.restype = k4a_result_t
    _k4a_playback_open.argtypes = (
        ctypes.POINTER(ctypes.c_char), ctypes.POINTER(k4a_playback_t),)

    return _k4a_playback_open(file_path, playback_handle)


def k4a_playback_close(playback_handle):
    _k4a_playback_close = record_dll.k4a_playback_close
    _k4a_playback_close.restype = None
    _k4a_playback_close.argtypes = (k4a_playback_t,)

    _k4a_playback_close(playback_handle)


def k4a_playback_get_raw_calibration(playback_handle, data, data_size):
    _k4a_playback_get_raw_calibration = (
        record_dll.k4a_playback_get_raw_calibration)
    _k4a_playback_get_raw_calibration.restype = k4a_buffer_result_t
    _k4a_playback_get_raw_calibration.argtypes = (
        k4a_playback_t, ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_size_t),)

    return _k4a_playback_get_raw_calibration(playback_handle, data, data_size)


def k4a_playback_get_calibration(playback_handle, calibration):
    _k4a_playback_get_calibration = record_dll.k4a_playback_get_calibration
    _k4a_playback_get_calibration.restype = k4a_result_t
    _k4a_playback_get_calibration.argtypes = (
        k4a_playback_t, ctypes.POINTER(k4a_calibration_t),)

    return _k4a_playback_get_calibration(playback_handle, calibration)


def k4a_playback_get_record_configuration(playback_handle, config):
    _k4a_playback_get_record_configuration = (
        record_dll.k4a_playback_get_record_configuration)
    _k4a_playback_get_record_configuration.restype = k4a_result_t
    _k4a_playback_get_record_configuration.argtypes = (
        k4a_playback_t, ctypes.POINTER(k4a_record_configuration_t),)

    return _k4a_playback_get_record_configuration(playback_handle, config)


def k4a_playback_check_track_exists(playback_handle, track_name):
    _k4a_playback_check_track_exists = (
        record_dll.k4a_playback_check_track_exists)
    _k4a_playback_check_track_exists.restype = ctypes.c_bool
    _k4a_playback_check_track_exists.argtypes = (
        k4a_playback_t, ctypes.POINTER(ctypes.c_char),)

    return _k4a_playback_check_track_exists(playback_handle, track_name)


def k4a_playback_get_track_count(playback_handle):
    _k4a_playback_get_track_count = record_dll.k4a_playback_get_track_count
    _k4a_playback_get_track_count.restype = ctypes.c_size_t
    _k4a_playback_get_track_count.argtypes = (k4a_playback_t,)

    return _k4a_playback_get_track_count(playback_handle)


def k4a_playback_get_track_name(
        playback_handle, track_index, track_name, track_name_size):
    _k4a_playback_get_track_name = record_dll.k4a_playback_get_track_name
    _k4a_playback_get_track_name.restype = k4a_buffer_result_t
    _k4a_playback_get_track_name.argtypes = (
        k4a_playback_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(ctypes.c_size_t),)

    return _k4a_playback_get_track_name(
        playback_handle, track_index, track_name, track_name_size)


def k4a_playbk4a_playback_track_is_builtinack_get_track_name(
        playback_handle, track_name):
    _k4a_playback_track_is_builtin = record_dll.k4a_playback_track_is_builtin
    _k4a_playback_track_is_builtin.restype = ctypes.c_bool
    _k4a_playback_track_is_builtin.argtypes = (
        k4a_playback_t, ctypes.POINTER(ctypes.c_char),)

    return _k4a_playback_track_is_builtin(playback_handle, track_name)


def k4a_playback_track_get_video_settings(
        playback_handle, track_name, video_settings):
    _k4a_playback_track_get_video_settings = (
        record_dll.k4a_playback_track_get_video_settings)
    _k4a_playback_track_get_video_settings.restype = k4a_result_t
    _k4a_playback_track_get_video_settings.argtypes = (
        k4a_playback_t, ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(k4a_record_video_settings_t),)

    return _k4a_playback_track_get_video_settings(
        playback_handle, track_name, video_settings)


def k4a_playback_track_get_codec_id(
        playback_handle, track_name, codec_id, codec_id_size):
    _k4a_playback_track_get_codec_id = (
        record_dll.k4a_playback_track_get_codec_id)
    _k4a_playback_track_get_codec_id.restype = k4a_buffer_result_t
    _k4a_playback_track_get_codec_id.argtypes = (
        k4a_playback_t, ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_size_t),)

    return _k4a_playback_track_get_codec_id(
        playback_handle, track_name, codec_id, codec_id_size)


def k4a_playback_track_get_codec_context(playback_handle, track_name,
                                         codec_context, codec_context_size):
    _k4a_playback_track_get_codec_context = (
        record_dll.k4a_playback_track_get_codec_context)
    _k4a_playback_track_get_codec_context.restype = k4a_buffer_result_t
    _k4a_playback_track_get_codec_context.argtypes = (
        k4a_playback_t, ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_size_t),)

    return _k4a_playback_track_get_codec_context(
        playback_handle, track_name, codec_context, codec_context_size)


def k4a_playback_get_tag(playback_handle, name, value, value_size):
    _k4a_playback_get_tag = record_dll.k4a_playback_get_tag
    _k4a_playback_get_tag.restype = k4a_buffer_result_t
    _k4a_playback_get_tag.argtypes = (
        k4a_playback_t, ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_size_t),)

    return _k4a_playback_get_tag(playback_handle, name, value, value_size)


def k4a_playback_set_color_conversion(playback_handle, target_format):
    _k4a_playback_set_color_conversion = (
        record_dll.k4a_playback_set_color_conversion)
    _k4a_playback_set_color_conversion.restype = k4a_result_t
    _k4a_playback_set_color_conversion.argtypes = (
        k4a_playback_t, k4a_image_format_t,)

    return _k4a_playback_set_color_conversion(playback_handle, target_format)


def k4a_playback_get_attachment(playback_handle, file_name, data, data_size):
    _k4a_playback_get_attachment = record_dll.k4a_playback_get_attachment
    _k4a_playback_get_attachment.restype = k4a_buffer_result_t
    _k4a_playback_get_attachment.argtypes = (
        k4a_playback_t, ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_size_t),)

    return _k4a_playback_get_attachment(
        playback_handle, file_name, data, data_size)


def k4a_playback_get_next_capture(playback_handle, capture_handle):
    _k4a_playback_get_next_capture = record_dll.k4a_playback_get_next_capture
    _k4a_playback_get_next_capture.restype = k4a_stream_result_t
    _k4a_playback_get_next_capture.argtypes = (
        k4a_playback_t, ctypes.POINTER(k4a_capture_t),)

    return _k4a_playback_get_next_capture(playback_handle, capture_handle)


def k4a_playback_get_previous_capture(playback_handle, capture_handle):
    _k4a_playback_get_previous_capture = (
        record_dll.k4a_playback_get_previous_capture)
    _k4a_playback_get_previous_capture.restype = k4a_stream_result_t
    _k4a_playback_get_previous_capture.argtypes = (
        k4a_playback_t, ctypes.POINTER(k4a_capture_t),)

    return _k4a_playback_get_previous_capture(playback_handle, capture_handle)


def k4a_playback_get_next_imu_sample(playback_handle, imu_sample):
    _k4a_playback_get_next_imu_sample = record_dll.k4a_playback_get_next_imu_sample
    _k4a_playback_get_next_imu_sample.restype = k4a_stream_result_t
    _k4a_playback_get_next_imu_sample.argtypes = (
        k4a_playback_t, ctypes.POINTER(k4a_imu_sample_t),)

    return _k4a_playback_get_next_imu_sample(playback_handle, imu_sample)


def k4a_playback_get_previous_imu_sample(playback_handle, imu_sample):
    _k4a_playback_get_previous_imu_sample = record_dll.k4a_playback_get_previous_imu_sample
    _k4a_playback_get_previous_imu_sample.restype = k4a_stream_result_t
    _k4a_playback_get_previous_imu_sample.argtypes = (
        k4a_playback_t, ctypes.POINTER(k4a_imu_sample_t),)

    return _k4a_playback_get_previous_imu_sample(playback_handle, imu_sample)


def k4a_playback_get_next_data_block(
        playback_handle, track_name, data_block_handle):
    _k4a_playback_get_next_data_block = (
        record_dll.k4a_playback_get_next_data_block)
    _k4a_playback_get_next_data_block.restype = k4a_stream_result_t
    _k4a_playback_get_next_data_block.argtypes = (
        k4a_playback_t, ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(k4a_playback_data_block_t),)

    return _k4a_playback_get_next_data_block(
        playback_handle, track_name, data_block_handle)


def k4a_playback_get_previous_data_block(
        playback_handle, track_name, data_block_handle):
    _k4a_playback_get_previous_data_block = (
        record_dll.k4a_playback_get_previous_data_block)
    _k4a_playback_get_previous_data_block.restype = k4a_stream_result_t
    _k4a_playback_get_previous_data_block.argtypes = (
        k4a_playback_t, ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(k4a_playback_data_block_t),)

    return _k4a_playback_get_previous_data_block(playback_handle, track_name,
                                                 data_block_handle)


def k4a_playback_data_block_get_device_timestamp_usec(data_block_handle):
    _k4a_playback_data_block_get_device_timestamp_usec = (
        record_dll.k4a_playback_data_block_get_device_timestamp_usec)
    _k4a_playback_data_block_get_device_timestamp_usec.restype = (
        ctypes.c_uint64)
    _k4a_playback_data_block_get_device_timestamp_usec.argtypes = (
        k4a_playback_data_block_t,)

    return _k4a_playback_data_block_get_device_timestamp_usec(
        data_block_handle)


def k4a_playback_data_block_get_buffer_size(data_block_handle):
    _k4a_playback_data_block_get_buffer_size = (
        record_dll.k4a_playback_data_block_get_buffer_size)
    _k4a_playback_data_block_get_buffer_size.restype = ctypes.c_size_t
    _k4a_playback_data_block_get_buffer_size.argtypes = (
        k4a_playback_data_block_t,)

    return _k4a_playback_data_block_get_buffer_size(data_block_handle)


def k4a_playback_data_block_get_buffer(data_block_handle):
    _k4a_playback_data_block_get_buffer = (
        record_dll.k4a_playback_data_block_get_buffer)
    _k4a_playback_data_block_get_buffer.restype = (
        ctypes.POINTER(ctypes.c_uint8))
    _k4a_playback_data_block_get_buffer.argtypes = (k4a_playback_data_block_t,)

    return _k4a_playback_data_block_get_buffer(data_block_handle)


def k4a_playback_data_block_release(data_block_handle):
    _k4a_playback_data_block_release = (
        record_dll.k4a_playback_data_block_release)
    _k4a_playback_data_block_release.restype = None
    _k4a_playback_data_block_release.argtypes = (k4a_playback_data_block_t,)

    return _k4a_playback_data_block_release(data_block_handle)


def k4a_playback_seek_timestamp(playback_handle, offset_usec, origin):
    _k4a_playback_seek_timestamp = record_dll.k4a_playback_seek_timestamp
    _k4a_playback_seek_timestamp.restype = k4a_result_t
    _k4a_playback_seek_timestamp.argtypes = (
        k4a_playback_t, ctypes.c_int64, k4a_playback_seek_origin_t,)

    return _k4a_playback_seek_timestamp(playback_handle, offset_usec, origin)


def k4a_playback_get_recording_length_usec(playback_handle):
    _k4a_playback_get_recording_length_usec = (
        record_dll.k4a_playback_get_recording_length_usec)
    _k4a_playback_get_recording_length_usec.restype = ctypes.c_uint64
    _k4a_playback_get_recording_length_usec.argtypes = (k4a_playback_t,)

    return _k4a_playback_get_recording_length_usec(playback_handle)


def VERIFY(result, error):
    if result != K4A_RESULT_SUCCEEDED:
        print(error)
        traceback.print_stack()
        sys.exit(1)
