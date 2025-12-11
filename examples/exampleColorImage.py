import cv2
import time

import pykinect_azure as pykinect


def main():
	# Initialize the library.
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	# Modify camera configuration.
	device_config = pykinect.default_configuration
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF
	# print(device_config)

	device = pykinect.start_device(config=device_config)
	cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
	FRAME_WINDOW = 600
	frame_count = 0
	start_time = time.perf_counter()
	while True:
		frame_count += 1
		capture = device.update()
		color_image = capture.get_color_image()
		cv2.imshow("Color Image", color_image)

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
