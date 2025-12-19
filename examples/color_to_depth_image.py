import cv2

import pykinect_azure as pykinect


def main():
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	device_config = pykinect.default_configuration
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	device_config.synchronized_images_only = True

	device = pykinect.start_device(config=device_config)
	transformation = device.transformation

	cv2.namedWindow("Transformed color image", cv2.WINDOW_NORMAL)
	frc = pykinect.FrameRateCalculator()
	frc.start()
	while True:
		capture = device.update()

		depth_image = capture.get_depth_image()
		depth_data = depth_image.to_numpy()
		depth_data = transformation.color_a_depth_image(depth_data)

		color_image = capture.get_color_image()
		transformed_color_image = transformation.color_image_to_depth_camera(
			depth_image, color_image)
		bgra_data = transformed_color_image.to_numpy()
		gray_view = bgra_data[:, :, 1]
		bgra_data = cv2.merge([gray_view, gray_view, gray_view])

		combined_image = cv2.addWeighted(
			bgra_data[:, :, :3], 0.7, depth_data, 0.3, 0)
		cv2.imshow("Transformed color image", combined_image)

		# Press q key to stop
		if cv2.waitKey(1) == ord("q"):
			break
		frc.update()
	del capture
	del device


if __name__ == "__main__":
	main()
