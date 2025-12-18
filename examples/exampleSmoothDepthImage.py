import cv2
import numpy as np
import time

import pykinect_azure as pykinect


def main():
	maximum_hole_size = 10

	# Initialize the library.
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	# Modify camera configuration.
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

	device = pykinect.start_device(config=device_config)
	cv2.namedWindow("Smoothed Depth Comparison", cv2.WINDOW_NORMAL)
	FRAME_WINDOW = 600
	frame_count = 0
	start_time = time.perf_counter()
	while True:
		frame_count += 1
		capture = device.update()
		raw_depth_image = capture.get_colored_depth_image()

		smooth_depth_color_image = capture.get_smooth_colored_depth_image(
			maximum_hole_size)
		comparison_image = np.concatenate(
			(raw_depth_image, smooth_depth_color_image), axis=1)
		comparison_image = cv2.putText(
			comparison_image, "Original", (180, 50),
			cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
		comparison_image = cv2.putText(
			comparison_image, "Smoothed", (670, 50),
			cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
		cv2.imshow("Smoothed Depth Comparison", comparison_image)

		# Press q key to stop.
		if cv2.waitKey(1) == ord('q'):  
			break

		if frame_count >= FRAME_WINDOW:
			end_time = time.perf_counter()
			elapsed_time = end_time - start_time
			fps = frame_count / elapsed_time
			print(f"FPS: {fps:.2f}")
			start_time = time.perf_counter()
			frame_count = 0
	# Manually deallocate the memory.
	del capture
	del device


if __name__ == "__main__":
	main()
