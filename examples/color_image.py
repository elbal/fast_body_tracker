import cv2

import pykinect_azure as pykinect


def main():
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	device_config = pykinect.Configuration()
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF

	device = pykinect.start_device(config=device_config)
	cv2.namedWindow("Color image", cv2.WINDOW_NORMAL)
	frc = pykinect.FrameRateCalculator()
	frc.start()
	while True:
		capture = device.update()
		image_object = capture.get_color_image_object()
		bgra_image = image_object.to_numpy()
		cv2.imshow("Color Image", bgra_image)

		# Press q key to stop.
		if cv2.waitKey(1) == ord("q"):
			break
		frc.update()
	del capture
	del device


if __name__ == "__main__":
	main()
