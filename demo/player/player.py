# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread

import os
import sys
import cv2
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate
from mmaction.apis import init_recognizer

from PySide6.QtCore import QStandardPaths, Qt, Slot
from PySide6.QtGui import QAction, QIcon, QKeySequence, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog,
                                QMainWindow, QStyle, QToolBar, QWidget,
                                QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox,
                                QLabel)
from PySide6.QtMultimedia import (QAudioOutput, QMediaFormat, QMediaPlayer)
from PySide6.QtMultimediaWidgets import QVideoWidget

from webcam import WebcamThread

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
# FONTCOLOR = (255, 255, 255)  # BGR, white
FONTCOLOR = (0, 0, 255)  # BGR, red
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]
AVI = "video/x-msvideo"  # AVI
MP4 = 'video/mp4'


def get_supported_mime_types():
    result = []
    for f in QMediaFormat().supportedFileFormats(QMediaFormat.Decode):
        mime_type = QMediaFormat(f).mimeType()
        result.append(mime_type.name())
    return result

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self._playlist = []
        self._playlist_index = -1
        self._audio_output = QAudioOutput()
        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._audio_output)
        self._player.errorOccurred.connect(self._player_error)

        self.is_webcam = False
        self.cap = None
        self.webcam_thread = WebcamThread(self)
        self.webcam_thread.frame_update.connect(self.update_webcam)

        self.__initUI()
        self._mime_types = []

    def __initUI(self):
        self.setWindowTitle("Action Recognition Viewer")

        ## menu & toolbar
        tool_bar = QToolBar()
        self.addToolBar(tool_bar)
        style = self.style()

        script_path = os.path.abspath(__file__)
        script_directory = os.path.dirname(script_path)
        directory_path = str(script_directory)

        file_menu = self.menuBar().addMenu("&File")
        icon = QIcon.fromTheme("document-open.png",
                               style.standardIcon(QStyle.SP_DirOpenIcon))
        open_action = QAction(icon, "&Open...", self,
                              shortcut=QKeySequence.Open, triggered=self.open)
        file_menu.addAction(open_action)
        tool_bar.addAction(open_action)
        icon = QIcon(directory_path + '\\webcam.png')
        webcam_action = QAction(icon, "&Webcam", self,
                              shortcut="Ctrl+W", triggered=self.capture)
        file_menu.addAction(webcam_action)
        tool_bar.addAction(webcam_action)
        icon = QIcon.fromTheme("application-exit.png",
                               style.standardIcon(QStyle.SP_TitleBarCloseButton))
        exit_action = QAction(icon, "E&xit", self,
                              shortcut="Ctrl+Q", triggered=self.close)
        file_menu.addAction(exit_action)

        play_menu = self.menuBar().addMenu("&Play")
        icon = QIcon.fromTheme("media-playback-start.png",
                               style.standardIcon(QStyle.SP_MediaPlay))
        self._play_action = tool_bar.addAction(icon, "Play")
        self._play_action.triggered.connect(self._player.play)
        play_menu.addAction(self._play_action)

        icon = QIcon.fromTheme("media-skip-backward-symbolic.svg",
                               style.standardIcon(QStyle.SP_MediaSkipBackward))
        self._previous_action = tool_bar.addAction(icon, "Previous")
        self._previous_action.triggered.connect(self.previous_clicked)
        play_menu.addAction(self._previous_action)

        icon = QIcon.fromTheme("media-playback-pause.png",
                               style.standardIcon(QStyle.SP_MediaPause))
        self._pause_action = tool_bar.addAction(icon, "Pause")
        self._pause_action.triggered.connect(self._player.pause)
        play_menu.addAction(self._pause_action)

        icon = QIcon.fromTheme("media-skip-forward-symbolic.svg",
                               style.standardIcon(QStyle.SP_MediaSkipForward))
        self._next_action = tool_bar.addAction(icon, "Next")
        self._next_action.triggered.connect(self.next_clicked)
        play_menu.addAction(self._next_action)

        icon = QIcon.fromTheme("media-playback-stop.png",
                               style.standardIcon(QStyle.SP_MediaStop))
        self._stop_action = tool_bar.addAction(icon, "Stop")
        self._stop_action.triggered.connect(self._ensure_stopped)
        play_menu.addAction(self._stop_action)

        ## video
        layout = QVBoxLayout()

        self._video_widget = QVideoWidget()
        self.setCentralWidget(self._video_widget)
        self._player.playbackStateChanged.connect(self.update_buttons)
        self._player.setVideoOutput(self._video_widget)
        self.update_buttons(self._player.playbackState())

        self._webcam_widget = QLabel(self)
        layout.addWidget(self._webcam_widget)

        layout.addWidget(self._video_widget)
        self.central_widget = QWidget()
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)

    def closeEvent(self, event):
        self._ensure_stopped()
        event.accept()

    @Slot()
    def open(self):
        self._ensure_stopped()
        file_dialog = QFileDialog(self)

        is_windows = sys.platform == 'win32'
        if not self._mime_types:
            self._mime_types = get_supported_mime_types()
            if (is_windows and AVI not in self._mime_types):
                self._mime_types.append(AVI)
            elif MP4 not in self._mime_types:
                self._mime_types.append(MP4)

        file_dialog.setMimeTypeFilters(self._mime_types)

        # default_mimetype = AVI if is_windows else MP4
        default_mimetype = MP4
        if default_mimetype in self._mime_types:
            file_dialog.selectMimeTypeFilter(default_mimetype)

        movies_location = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
        file_dialog.setDirectory(movies_location)
        if file_dialog.exec() == QDialog.Accepted:
            url = file_dialog.selectedUrls()[0]
            self._playlist.append(url)
            self._playlist_index = len(self._playlist) - 1
            self._player.setSource(url)
            self._player.play()
    
    @Slot()
    def capture(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap:
            self.__show_error("No Camera")
            return
        if not self.is_webcam:
            self.is_webcam = True
            self.webcam_thread.start()
        else:
            self.is_webcam = False
            self.webcam_thread.stop()

    @Slot()
    def _ensure_stopped(self):
        if self._player.playbackState() != QMediaPlayer.StoppedState:
            self._player.stop()

    @Slot()
    def previous_clicked(self):
        # Go to previous track if we are within the first 5 seconds of playback
        # Otherwise, seek to the beginning.
        if self._player.position() <= 5000 and self._playlist_index > 0:
            self._playlist_index -= 1
            self._playlist.previous()
            self._player.setSource(self._playlist[self._playlist_index])
        else:
            self._player.setPosition(0)

    @Slot()
    def next_clicked(self):
        if self._playlist_index < len(self._playlist) - 1:
            self._playlist_index += 1
            self._player.setSource(self._playlist[self._playlist_index])

    @Slot("QMediaPlayer::PlaybackState")
    def update_buttons(self, state):
        media_count = len(self._playlist)
        self._play_action.setEnabled(media_count > 0
            and state != QMediaPlayer.PlayingState)
        self._pause_action.setEnabled(state == QMediaPlayer.PlayingState)
        self._stop_action.setEnabled(state != QMediaPlayer.StoppedState)
        self._previous_action.setEnabled(self._player.position() > 0)
        self._next_action.setEnabled(media_count > 1)

    def show_status_message(self, message):
        self.statusBar().showMessage(message, 5000)

    @Slot("QMediaPlayer::Error", str)
    def _player_error(self, error, error_string):
        print(error_string, file=sys.stderr)
        self.show_status_message(error_string)

    def __show_error(self, message):
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Warning)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.exec_()

    def update_webcam(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        q_image = QImage(frame.data, w, h, c * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self._webcam_widget.setPixmap(pixmap)
        self._webcam_widget.setAlignment(Qt.AlignCenter)

    def closeEvent(self, event):
        self.webcam_thread.stop()
        event.accept()
        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    available_geometry = main_win.screen().availableGeometry()
    main_win.resize(available_geometry.width() / 3,
                    available_geometry.height() / 2)
    main_win.show()
    sys.exit(app.exec())
