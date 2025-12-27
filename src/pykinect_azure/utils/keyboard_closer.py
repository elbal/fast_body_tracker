import threading
from pynput import keyboard


class KeyboardCloser:
    def __init__(self):
        self.stop_event = threading.Event()
        self.listener = keyboard.Listener(on_press=self.on_press)

    def on_press(self, key):
        # Press q key to stop.
        try:
            if getattr(key, "char", None) == "q":
                self.stop_event.set()
                return False
        except AttributeError:
            pass

    def start(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
