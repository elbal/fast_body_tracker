import cv2
import numpy as np
import threading
import queue

import pykinect_azure as pykinect


def capture_thread(device, q, stop_event):
	dfa = pykinect.DroppedFramesAlert()
	while not stop_event.is_set():
		capture = device.update()
		if q.full():
			dfa.update()
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

	cv2.namedWindow("Depth image", cv2.WINDOW_NORMAL)
	depth_8bit_image = np.zeros((512, 512), dtype=np.uint8)
	colorized_image = np.zeros((512, 512, 3), dtype=np.uint8)

	frc = pykinect.FrameRateCalculator()

	t.start()
	frc.start()
	while True:
		capture = q.get()
		image_object = capture.get_depth_image_object()
		depth_image = image_object.to_numpy()
		cv2.convertScaleAbs(depth_image, alpha=0.08, dst=depth_8bit_image)
		cv2.applyColorMap(
			depth_8bit_image, cv2.COLORMAP_CIVIDIS, dst=colorized_image)
		cv2.imshow("Depth image", colorized_image)

		if cv2.waitKey(1) == ord("q"):
			break
		frc.update()

	cv2.destroyAllWindows()
	stop_event.set()
	t.join()
	del device


if __name__ == "__main__":
	main()
