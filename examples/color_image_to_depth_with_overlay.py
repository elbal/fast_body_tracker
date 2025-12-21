import cv2
import numpy as np
import threading
import queue
import pykinect_azure as pykinect


def capture_thread(device, q, stop_event):
	while not stop_event.is_set():
		capture = device.update()
		if q.full():
			try:
				q.get_nowait()
			except queue.Empty:
				pass
		q.put(capture)


def main():
	pykinect.initialize_libraries()

	device_config = pykinect.Configuration()
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	device_config.synchronized_images_only = True

	device = pykinect.start_device(config=device_config)
	transformation = device.transformation

	q = queue.Queue(maxsize=10)
	stop_event = threading.Event()
	t = threading.Thread(target=capture_thread, args=(device, q, stop_event))
	t.start()

	cv2.namedWindow("Transformed color image", cv2.WINDOW_NORMAL)
	frc = pykinect.FrameRateCalculator()
	frc.start()

	depth_8bit_image = np.zeros((512, 512), dtype=np.uint8)
	colorized_image = np.zeros((512, 512, 3), dtype=np.uint8)
	combined_image = np.zeros((512, 512, 3), dtype=np.uint8)

	while True:
		capture = q.get()

		depth_image_object = capture.get_depth_image_object()
		depth_image = depth_image_object.to_numpy()
		cv2.convertScaleAbs(depth_image, alpha=0.05, dst=depth_8bit_image)
		cv2.applyColorMap(
			depth_8bit_image, cv2.COLORMAP_CIVIDIS, dst=colorized_image)

		color_image_object = capture.get_color_image_object()
		transformed_image_object = transformation.color_image_to_depth_camera(
			depth_image_object, color_image_object)

		bgra_image = transformed_image_object.to_numpy()
		gray_view = bgra_image[:, :, 1]
		bgra_gray_merged = cv2.merge([gray_view, gray_view, gray_view])

		cv2.addWeighted(
			bgra_gray_merged, 0.7, colorized_image, 0.3, 0, dst=combined_image)
		cv2.imshow("Transformed color image", combined_image)

		if cv2.waitKey(1) == ord("q"):
			break
		frc.update()

	cv2.destroyAllWindows()
	stop_event.set()
	t.join()
	del device


if __name__ == "__main__":
	main()
