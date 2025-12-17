import cv2
import numpy as np

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
	cv2.namedWindow("Infrared image", cv2.WINDOW_NORMAL)
	frc = pykinect.FrameRateCalculator()
	frc.start()
	while True:
		capture = device.update()
		image = capture.get_ir_image()
		ir_data = image.to_numpy()
		ir_data = ir_data / 2000.0 * 255.0
		ir_data = np.clip(ir_data, 0, 255).astype(np.uint8)
		ir_data = cv2.applyColorMap(ir_data, cv2.COLORMAP_CIVIDIS)
		cv2.imshow("Infrared image", ir_data)

		# Press q key to stop.
		if cv2.waitKey(1) == ord("q"):
			break
		frc.update()
	# Manually deallocate the memory.
	del capture
	del device


if __name__ == "__main__":
	main()
