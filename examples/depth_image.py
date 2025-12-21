import cv2
import numpy as np

import pykinect_azure as pykinect


def main():
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	device_config = pykinect.Configuration()
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

	device = pykinect.start_device(config=device_config)
	cv2.namedWindow("Depth image", cv2.WINDOW_NORMAL)
	frc = pykinect.FrameRateCalculator()
	frc.start()
	depth_8bit_image = np.zeros((512, 512), dtype=np.uint8)
	colorized_image = np.zeros((512, 512, 3), dtype=np.uint8)
	while True:
		capture = device.update()
		image_object = capture.get_depth_image_object()
		depth_image = image_object.to_numpy()
		cv2.convertScaleAbs(
			depth_image, alpha=0.08, dst=depth_8bit_image)
		cv2.applyColorMap(
			depth_8bit_image, cv2.COLORMAP_TURBO, dst=colorized_image)

		cv2.imshow("Depth image", colorized_image)

		# Press q key to stop.
		if cv2.waitKey(1) == ord("q"):
			break
		frc.update()
	del capture
	del device


if __name__ == "__main__":
	main()
