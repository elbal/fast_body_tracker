import pykinect_azure as pykinect
from pykinect_azure import PointCloudVisualizer, KeyboardCloser


def main():
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	device_config = pykinect.Configuration()
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

	device = pykinect.start_device(config=device_config)
	transformation = device.transformation

	visualizer = PointCloudVisualizer()
	keyboard_closer = KeyboardCloser()
	keyboard_closer.start()
	while not keyboard_closer.stop_event.is_set():
		capture = device.update()
		depth_image_object = capture.get_depth_image_object()
		point_cloud_object = transformation.depth_image_to_point_cloud(
			depth_image_object,
			calibration_type=pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		point_cloud = point_cloud_object.to_numpy()
		visualizer(point_cloud)
	del capture
	del device


if __name__ == "__main__":
	main()
