import cv2

import pykinect_azure as pykinect


def main():
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

	device = pykinect.start_device(config=device_config)
	cv2.namedWindow("Infrared image", cv2.WINDOW_NORMAL)
	frc = pykinect.FrameRateCalculator()
	frc.start()
	ir_scale_factor = 255.0 / 500.0  # To improve visualization.
	while True:
		capture = device.update()
		image = capture.get_ir_image()
		ir_data = image.to_numpy()
		ir_data = cv2.convertScaleAbs(ir_data, alpha=ir_scale_factor)
		cv2.imshow("Infrared image", ir_data)

		# Press q key to stop.
		if cv2.waitKey(1) == ord("q"):
			break
		frc.update()
	del capture
	del device


if __name__ == "__main__":
	main()
