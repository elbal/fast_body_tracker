import time


class FrameRateCalculator:
    def __init__(self):
        self.frame_window = 600
        self.frame_count = 0
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def update(self):
        self.frame_count += 1
        if self.frame_count >= self.frame_window:
            end_time = time.perf_counter()
            elapsed_time = end_time - self.start_time
            fps = self.frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            self.start_time = time.perf_counter()
            self.frame_count = 0
