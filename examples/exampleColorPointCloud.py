import threading
from pynput import keyboard

import pykinect_azure as pykinect
from pykinect_azure.k4a import PointCloudVisualizer


def main():
	# Initialize the library.
	# If the library is not found, add the library path as argument.
	pykinect.initialize_libraries()

	# Modify camera configuration.
	device_config = pykinect.default_configuration
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
	device_config.synchronized_images_only = True
	# print(device_config)

	device = pykinect.start_device(config=device_config)
	visualizer = PointCloudVisualizer()
	stop_event = threading.Event()

	def on_press(key):
		# Press q key to stop.
		try:
			if getattr(key, "char", None) == "q":
				stop_event.set()
				return False
		except AttributeError:
			pass

	listener = keyboard.Listener(on_press=on_press)
	listener.start()

	while not stop_event.is_set():
		capture = device.update()
		points = capture.get_transformed_pointcloud()
		color_image = capture.get_color_image()

		visualizer(points, color_image)


if __name__ == "__main__":
	main()
