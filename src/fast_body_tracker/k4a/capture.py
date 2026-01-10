from ._k4a_types import k4a_capture_t
from . import _k4a
from .image import Image


class Capture:
    def __init__(self, capture_handle: k4a_capture_t):
        self._handle = capture_handle

    def __del__(self):
        if self._handle:
            _k4a.K4aLib.k4a_capture_release(self._handle)

    def handle(self) -> k4a_capture_t:
        return self._handle

    def get_color_image_object(self) -> Image:
        image_handle = _k4a.K4aLib.k4a_capture_get_color_image(self._handle)

        return Image(image_handle)

    def get_depth_image_object(self) -> Image:
        image_handle = _k4a.K4aLib.k4a_capture_get_depth_image(self._handle)

        return Image(image_handle)

    def get_ir_image_object(self) -> Image:
        image_handle = _k4a.K4aLib.k4a_capture_get_ir_image(self._handle)

        return Image(image_handle)
