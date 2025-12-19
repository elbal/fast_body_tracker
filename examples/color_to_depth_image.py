import cv2

import pykinect_azure as pykinect


def main():
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	device_config = pykinect.Configuration()
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

		depth_image_object = capture.get_depth_image_object()
		depth_image = depth_image_object.to_numpy()
		depth_image = transformation.color_a_depth_image(depth_image)

		color_image_object = capture.get_color_image_object()
		transformed_image_object = transformation.color_image_to_depth_camera(
			depth_image_object, color_image_object)
		bgra_image = transformed_image_object.to_numpy()
		gray_view = bgra_image[:, :, 1]
		bgra_image = cv2.merge([gray_view, gray_view, gray_view])

		combined_image = cv2.addWeighted(
			bgra_image[:, :, :3], 0.7, depth_image, 0.3, 0)
		cv2.imshow("Transformed color image", combined_image)

		# Press q key to stop
		if cv2.waitKey(1) == ord("q"):
			break
		frc.update()
	del capture
	del device


if __name__ == "__main__":
	main()
