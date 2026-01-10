from ._k4a_types import k4a_device_configuration_t
from . import k4a_const


class Configuration:
    def __init__(self):
        self.color_format = k4a_const.K4A_IMAGE_FORMAT_COLOR_MJPG
        self.color_resolution = k4a_const.K4A_COLOR_RESOLUTION_720P
        self.depth_mode = k4a_const.K4A_DEPTH_MODE_WFOV_2X2BINNED
        self.camera_fps = k4a_const.K4A_FRAMES_PER_SECOND_30
        self.synchronized_images_only = False
        self.depth_delay_off_color_usec = 0
        self.wired_sync_mode = k4a_const.K4A_WIRED_SYNC_MODE_STANDALONE
        self.subordinate_delay_off_master_usec = 0
        self.disable_streaming_indicator = False

        self._handle = self._on_value_change()

    def handle(self):
        return self._handle

    def __setattr__(self, name: str, value: int | bool):
        if hasattr(self, name):
            if name != "_handle":
                if self.__dict__[name] != value:
                    self.__dict__[name] = value
                    self._handle = self._on_value_change()
            else:
                self.__dict__[name] = value
        else:
            self.__dict__[name] = value

    def __str__(self) -> str:
        message = (
            "Device configuration: \n"
            f"\tcolor_format: {self.color_format} "
            f"\n\t(0:JPG, 1:NV12, 2:YUY2, 3:BGRA32)\n\n"
            f"\tcolor_resolution: {self.color_resolution} "
            f"\n\t(0:OFF, 1:720p, 2:1080p, 3:1440p, "
            f"4:1536p, 5:2160p, 6:3072p)\n\n"
            f"\tdepth_mode: {self.depth_mode} "
            f"\n\t(0:OFF, 1:NFOV_2X2BINNED, 2:NFOV_UNBINNED, "
            f"3:WFOV_2X2BINNED, 4:WFOV_UNBINNED, 5:Passive IR)\n\n"
            f"\tcamera_fps: {self.camera_fps} \n\t(0:5 FPS, 1:15 FPS, "
            f"2:30 FPS)\n\n"
            f"\tsynchronized_images_only: {self.synchronized_images_only} \n\t"
            f"(True of False). "
            f"Drop images if the color and depth are not synchronized\n\n"
            f"\tdepth_delay_off_color_usec: "
            f"{self.depth_delay_off_color_usec} us. \n\t"
            f"Delay between the color image and the depth image\n\n"
            f"\twired_sync_mode: {self.wired_sync_mode}\n\t"
            f"(0:Standalone mode, 1:Master mode, 2:Subordinate mode)\n\n"
            f"\tsubordinate_delay_off_master_usec: "
            f"{self.subordinate_delay_off_master_usec} us.\n\t"
            f"The external synchronization timing.\n\n"
            f"\tdisable_streaming_indicator: "
            f"{self.disable_streaming_indicator} \n\t"
            f"(True or False). Streaming indicator automatically turns on "
            f"when the color or depth camera's are in use.\n\n")

        return message

    def _on_value_change(self) -> k4a_device_configuration_t:
        configuration_handle = k4a_device_configuration_t(
            self.color_format, self.color_resolution, self.depth_mode,
            self.camera_fps, self.synchronized_images_only,
            self.depth_delay_off_color_usec, self.wired_sync_mode,
            self.subordinate_delay_off_master_usec,
            self.disable_streaming_indicator)

        return configuration_handle
