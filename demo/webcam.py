from PySide6.QtCore import QThread, Signal

class WebcamThread(QThread):
    frame_update = Signal(object)

    def __init__(self, video_player):
        super().__init__()
        self.stopped = False
        self.video_player = video_player

    def run(self):
        while not self.stopped:
            if self.video_player.cap.isOpened():
                ret, frame = self.video_player.cap.read()
                if ret:
                    self.frame_update.emit(frame)
                else:
                    self.video_player.cap.release()
                    self.video_player.is_webcam = False
                    self.stop()

    def stop(self):
        self.stopped = True
