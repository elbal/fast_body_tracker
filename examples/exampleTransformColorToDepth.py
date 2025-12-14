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
	device_config.synchronized_images_only = True
	# print(device_config)

	device = pykinect.start_device(config=device_config)
	cv2.namedWindow("Transformed Color Image", cv2.WINDOW_NORMAL)
	FRAME_WINDOW = 600
	frame_count = 0
	start_time = time.perf_counter()
	while True:
		frame_count += 1
		capture = device.update()
		color_image = capture.get_transformed_color_image()
		depth_image = capture.get_colored_depth_image()
		combined_image = cv2.addWeighted(
			color_image[:, :, :3], 0.7, depth_image, 0.3, 0)
		cv2.imshow("Transformed Color Image", combined_image)

		# Press q key to stop
		if cv2.waitKey(1) == ord("q"):
			break

		if frame_count >= FRAME_WINDOW:
			end_time = time.perf_counter()
			elapsed_time = end_time - start_time
			fps = frame_count / elapsed_time
			print(f"FPS: {fps:.2f}")
			start_time = time.perf_counter()
			frame_count = 0


if __name__ == "__main__":
	main()
