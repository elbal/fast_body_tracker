import cv2
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
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

	device = pykinect.start_device(config=device_config)

	q = queue.Queue(maxsize=10)
	stop_event = threading.Event()
	t = threading.Thread(target=capture_thread, args=(device, q, stop_event))
	t.start()

	cv2.namedWindow("Infrared image", cv2.WINDOW_NORMAL)
	frc = pykinect.FrameRateCalculator()
	frc.start()

	ir_scale_factor = 255.0 / 500.0

	while True:
		capture = q.get()
		image_object = capture.get_ir_image_object()

		ir_image = image_object.to_numpy()
		ir_image = cv2.convertScaleAbs(ir_image, alpha=ir_scale_factor)
		cv2.imshow("Infrared image", ir_image)

		if cv2.waitKey(1) == ord("q"):
			break
		frc.update()

	cv2.destroyAllWindows()
	stop_event.set()
	t.join()
	del device


if __name__ == "__main__":
	main()