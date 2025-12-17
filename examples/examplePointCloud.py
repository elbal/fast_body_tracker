import pykinect_azure as pykinect
from pykinect_azure import PointCloudVisualizer, KeyboardCloser


def main():
	# Initialize the library.
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

	device = pykinect.start_device(config=device_config)
	visualizer = PointCloudVisualizer()

	keyboard_closer = KeyboardCloser()
	keyboard_closer.start()
	while not keyboard_closer.stop_event.is_set():
		capture = device.update()
		image = capture.get_depth_image()
		point_cloud = device.transformation.depth_image_to_point_cloud(image)
		# Warning, do not delete the object before plotting or the data might
		# disappear.
		points = point_cloud.to_numpy()
		visualizer(points)
	# Manually deallocate the memory.
	del capture
	del device


if __name__ == "__main__":
	main()
