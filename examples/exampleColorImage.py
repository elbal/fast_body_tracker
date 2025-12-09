import cv2

import pykinect_azure as pykinect

if __name__ == "__main__":
	# Initialize the library.
	# If the library is not found add the library path as argument
	pykinect.initialize_libraries()

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	# print(device_config)

	device = pykinect.start_device(config=device_config)
	cv2.namedWindow('Color Image', cv2.WINDOW_NORMAL)
	while True:
		capture = device.update()
		color_image = capture.get_color_image()
		cv2.imshow("Color Image", color_image)
		
		# Press q key to stop
		if cv2.waitKey(1) == ord('q'): 
			break
