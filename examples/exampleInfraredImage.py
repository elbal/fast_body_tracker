import cv2
import numpy as np
import time

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
	cv2.namedWindow("Infrared Image", cv2.WINDOW_NORMAL)
	FRAME_WINDOW = 600
	frame_count = 0
	start_time = time.perf_counter()
	while True:
		frame_count += 1
		capture = device.update()
		ir_image = capture.get_ir_image()
		ir_image = ir_image / 2000.0 * 255.0
		ir_image = np.clip(ir_image, 0, 255).astype(np.uint8)
		ir_image = cv2.applyColorMap(ir_image, cv2.COLORMAP_CIVIDIS)

		cv2.imshow("Infrared Image", ir_image)

		if frame_count >= FRAME_WINDOW:
			end_time = time.perf_counter()
			elapsed_time = end_time - start_time
			fps = frame_count / elapsed_time
			print(f"FPS: {fps:.2f}")
			start_time = time.perf_counter()
			frame_count = 0
		
		# Press q key to stop.
		if cv2.waitKey(1) == ord("q"):
			break
	# Manually deallocate the memory.
	del capture
	del device


if __name__ == "__main__":
	main()
