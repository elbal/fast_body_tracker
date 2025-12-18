import cv2

import pykinect_azure as pykinect


def main():
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	device_config = pykinect.default_configuration
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
	device_config.synchronized_images_only = True

	device = pykinect.start_device(config=device_config)
	transformation = device.transformation

	cv2.namedWindow("Transformed depth image", cv2.WINDOW_NORMAL)
	frc = pykinect.FrameRateCalculator()
	frc.start()
	while True:
		capture = device.update()

		color_image = capture.get_color_image()
		bgra_data = color_image.to_numpy()

		depth_image = capture.get_depth_image()
		transformed_depth_image = transformation.depth_image_to_color_camera(
			depth_image)
		depth_data = transformed_depth_image.to_numpy()
		depth_data = transformation.color_a_depth_image(depth_data)

		combined_data = cv2.addWeighted(
			bgra_data[:, :, :3], 0.7, depth_data, 0.3, 0)
		cv2.imshow("Transformed depth image", combined_data)
		
		# Press q key to stop.
		if cv2.waitKey(1) == ord("q"):
			break
		frc.update()
	del capture
	del device


if __name__ == "__main__":
	main()
