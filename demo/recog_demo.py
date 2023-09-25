import sys
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.cap = None
        self.is_playing = False

        self.video_thread = VideoThread(self)
        self.video_thread.frame_update.connect(self.display_frame)

    def initUI(self):
        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.label = QLabel(self)
        layout.addWidget(self.label)

        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)
        layout.addWidget(self.select_button)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_video)
        layout.addWidget(self.play_button)

        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)

    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)", options=options)
        if filepath:
            self.cap = cv2.VideoCapture(filepath)

    def play_video(self):
        if not self.cap:
            return

        if not self.is_playing:
            self.is_playing = True
            self.play_button.setText("Pause")
            self.video_thread.start()
        else:
            self.is_playing = False
            self.play_button.setText("Play")
            self.video_thread.stop()

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignCenter)

class VideoThread(QThread):
    frame_update = pyqtSignal(object)

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
                    self.video_player.is_playing = False
                    self.video_player.play_button.setText("Play")
                    self.stop()

    def stop(self):
        self.stopped = True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoPlayer()
    window.show()
    sys.exit(app.exec_())
