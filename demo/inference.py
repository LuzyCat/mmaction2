import time
from PySide6.QtCore import QThread, Signal, Slot


class InferenceThread(QThread):
    infResult = Signal(float)

    def __init__(self, inference_fps):
        super().__init__()
        self.stopped = False
        self.inference_fps = inference_fps

    def run(self):
        cur_time = time.time()
        while not self.stopped:
            self.infResult.emit(cur_time)
            if self.inference_fps > 0:
                sleep_time = 1 / self.inference_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()

    def stop(self):
        self.stopped = True

