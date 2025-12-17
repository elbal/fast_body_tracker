import cv2

import pykinect_azure as pykinect


def main():
	# Initialize the library.
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	# Modify camera configuration.
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
	# print(device_config)

	device = pykinect.start_device(config=device_config)
	cv2.namedWindow("Depth image", cv2.WINDOW_NORMAL)
	frc = pykinect.FrameRateCalculator()
	frc.start()
	while True:
		capture = device.update()
		image = capture.get_depth_image()
		depth_data = image.to_numpy()
		depth_data = device.transformation.color_depth_image(depth_data)
		cv2.imshow("Depth image", depth_data)

		# Press q key to stop.
		if cv2.waitKey(1) == ord("q"):
			break
		frc.update()
	# Manually deallocate the memory.
	del capture
	del device


if __name__ == "__main__":
	main()
