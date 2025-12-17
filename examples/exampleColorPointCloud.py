import pykinect_azure as pykinect
from pykinect_azure import PointCloudVisualizer, KeyboardCloser


def main():
	# Initialize the library.
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	device_config = pykinect.default_configuration
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
	device_config.synchronized_images_only = True

	device = pykinect.start_device(config=device_config)
	transformation = device.transformation
	visualizer = PointCloudVisualizer()

	keyboard_closer = KeyboardCloser()
	keyboard_closer.start()
	while not keyboard_closer.stop_event.is_set():
		capture = device.update()
		color_image = capture.get_color_image()
		bgra_data = color_image.to_numpy()
		depth_image = capture.get_depth_image()
		transformed_depth_image = transformation.depth_image_to_color_camera(
			depth_image)
		point_cloud = transformation.depth_image_to_point_cloud(
			transformed_depth_image,
			calibration_type=pykinect.K4A_CALIBRATION_TYPE_COLOR)
		# Warning, do not delete point_cloud object before plotting or the data
		# might disappear.
		points = point_cloud.to_numpy()
		visualizer(points, bgra_data)
	# Manually deallocate the memory.
	del capture
	del device


if __name__ == "__main__":
	main()
