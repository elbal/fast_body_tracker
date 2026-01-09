import cv2
import time

import pykinect_azure as pykinect

if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries(track_body=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)
    calibration = device.calibration

    # Start body tracker
    tracker_config = pykinect.TrackerConfiguration()
    tracker = pykinect.start_body_tracker(calibration=calibration,
                                          tracker_configuration=tracker_config)

    cv2.namedWindow('Depth image with skeleton', cv2.WINDOW_NORMAL)
    FRAME_WINDOW = 600
    frame_count = 0
    start_time = time.perf_counter()
    while True:
        frame_count += 1

        # Get capture
        capture = device.update()

        # Get body tracker frame
        frame = tracker.update()

        # Get the color depth image from the capture
        depth_color_image = capture.get_colored_depth_image()

        # Get the colored body segmentation
        body_image_color = frame.get_segmentation_image_object()

        # Combine both images
        combined_image = cv2.addWeighted(depth_color_image, 0.6,
                                         body_image_color, 0.4, 0)

        # Draw the skeletons
        combined_image = frame.draw_bodies(combined_image)

        # Overlay body segmentation on depth image
        cv2.imshow('Depth image with skeleton', combined_image)

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break

        if frame_count >= FRAME_WINDOW:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            start_time = time.perf_counter()
            frame_count = 0
