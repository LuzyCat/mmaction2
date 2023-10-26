# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import cv2
import torch
import time
import argparse
import os.path as osp
import numpy as np
from operator import itemgetter
from collections import deque
from threading import Thread
# from typing import Optional, Tuple
# mmengine
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.visualization import ActionVisualizer
# Qt
from PySide6.QtCore import QStandardPaths, Qt, Slot, QRect
from PySide6.QtGui import QAction, QIcon, QKeySequence, QImage, QPixmap, QFontDatabase, QFont
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog,
                                QMainWindow, QStyle, QToolBar, QWidget,
                                QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox,
                                QLabel, QTextEdit, QMenu)
from PySide6.QtMultimedia import (QAudioOutput, QMediaFormat, QMediaPlayer)
from PySide6.QtMultimediaWidgets import QVideoWidget

from webcam import WebcamThread
from inference import InferenceThread

AVI = "video/x-msvideo"  # AVI
MP4 = 'video/mp4'

MODEL = None
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]


def get_supported_mime_types():
    result = []
    for f in QMediaFormat().supportedFileFormats(QMediaFormat.Decode):
        mime_type = QMediaFormat(f).mimeType()
        result.append(mime_type.name())
    return result

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 player demo')
    parser.add_argument('--config', default='demo/demo_configs/slowfast_act3d_video_infer.py',
                        help='test config file path')
    parser.add_argument('--checkpoint', default='checkpoints/slowfast_r50_8xb8-4x16x1-256e_act3d.pth',
                        help='checkpoint file/url')
    parser.add_argument('--label', default='tools/data/ETRI-Activity3D/label_map_ETRI-Activity3D.txt',
                        help='label file')
    parser.add_argument('--kor_label', default='tools/data/ETRI-Activity3D/label_map_ETRI-Activity3D_kor.txt',
                        help='label file(Korean)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--fps',
        default=30,
        type=int,
        help='specify fps value of the output video when using rawframes to '
        'generate file')
    parser.add_argument(
        '--font-scale',
        default=None,
        type=float,
        help='font scale of the text in output video')
    parser.add_argument(
        '--font-color',
        default='white',
        help='font color of the text in output video')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    parser.add_argument(
        '--average-size',
        type=int,
        default=5,
        help='number of latest clips to be averaged for prediction')
    parser.add_argument(
        '--inference-fps',
        type=int,
        default=5,
        help='Set upper bound FPS value of model inference')
    parser.add_argument(
        '--target-resolution',
        nargs=2,
        default=None,
        type=int,
        help='Target resolution (w, h) for resizing the frames when using a '
        'video as input. If either dimension is set to -1, the frames are '
        'resized by keeping the existing aspect ratio')
    parser.add_argument('--out-filename', default=None, help='output filename')
    args = parser.parse_args()
    return args


class MainWindow(QMainWindow):

    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.__initialize()

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
        self.inference_thread = InferenceThread(self.inference_fps)
        self.inference_thread.infResult.connect(self.update_result)
        self.frame_queue = deque(maxlen=self.sample_length)
        self.result_queue = deque(maxlen=1)

        self.__initUI()
        self._mime_types = []

    def __initialize(self):
        self.average_size = self.args.average_size
        self.threshold = self.args.threshold
        self.inference_fps = self.args.inference_fps
        self.device = torch.device(self.args.device)

        cfg = Config.fromfile(self.args.config)
        if self.args.cfg_options is not None:
            cfg.merge_from_dict(self.args.cfg_options)
        
        self.model = init_recognizer(cfg, self.args.checkpoint, device=self.args.device)
        self.data = dict(img_shape=None, modality='RGB', label=-1)

        cfg = self.model.cfg
        self.sample_length = 32
        pipeline = cfg.test_pipeline
        pipeline_ = pipeline.copy()
        for step in pipeline:
            if 'SampleFrames' in step['type']:
                self.sample_length = step['clip_len'] * step['num_clips']
                self.data['num_clips'] = step['num_clips']
                self.data['clip_len'] = step['clip_len']
                pipeline_.remove(step)
            if step['type'] in EXCLUED_STEPS:
                pipeline_.remove(step)
        self.pipeline = Compose(pipeline_)
        # eng
        labels = open(self.args.label,encoding='UTF-8').readlines()
        self.labels = [x.strip() for x in labels]
        # kor
        kor_labels = open(self.args.kor_label,encoding='UTF-8').readlines()
        self.kor_labels = [x.strip() for x in kor_labels]

    def __initUI(self):
        self.setWindowTitle("Action Recognition Viewer")
        self.setGeometry(100, 100, 1500, 1000)

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

        main_layout = QHBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        ## video
        v_layout = QVBoxLayout()
        v_layout.setGeometry(QRect(0, 0, 800, 1000))
        self.player_widget = QWidget()
        self.player_widget.setMinimumWidth(800)
        self.player_widget.setLayout(v_layout)

        self._video_widget = QVideoWidget(self.player_widget)
        self._video_widget.setMinimumHeight(600)
        self._video_widget.setMaximumHeight(800)
        # self.setCentralWidget(self._video_widget)
        self._player.playbackStateChanged.connect(self.update_buttons)
        self._player.setVideoOutput(self._video_widget)
        self._player.mediaStatusChanged.connect(self.on_media_status_changed)
        self.update_buttons(self._player.playbackState())

        self._webcam_widget = QLabel(self.player_widget)
        v_layout.addWidget(self._video_widget)
        v_layout.addWidget(self._webcam_widget)
        
        main_layout.addWidget(self.player_widget)

        ## result
        right_layout = QVBoxLayout()
        self.rightWidget = QWidget()
        self.rightWidget.setMinimumWidth(500)
        # self.rightWidget.setMaximumHeight(500)
        self.rightWidget.setLayout(right_layout)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_video)
        right_layout.addWidget(self.process_button)

        self.dropdown_menu = QMenu(self.process_button)
        self.process_button.setMenu(self.dropdown_menu)

        action1 = QAction("Video", self)
        action1.triggered.connect(self.process_video)
        self.dropdown_menu.addAction(action1)

        action2 = QAction("Webcam", self)
        action2.triggered.connect(self.process_webcam)
        self.dropdown_menu.addAction(action2)

        self.result_panel = QTextEdit(self)
        self.result_panel.setMinimumHeight(480)
        self.result_panel.setMaximumHeight(480)
        self.result_panel.setText('Result')
        right_layout.addWidget(self.result_panel)

        self.result_label = QLabel(self)
        self.result_label.setMaximumHeight(100)
        self.result_label.setText("-")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setFont(QFont("Noto Sans KR", 20, QFont.Bold))
        right_layout.addWidget(self.result_label)
        right_layout.setAlignment(Qt.AlignTop)

        main_layout.addWidget(self.rightWidget)

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
            self.video_path = url.toLocalFile()
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
    
    def on_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia:
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
        self.frame_queue.append(np.array(frame[:, :, ::-1]))
    
    def process_video(self):
        self.result_panel.clear()
        self.result_label.setText("-")

        try:
            # recognize action
            pred_result = inference_recognizer(self.model, self.video_path)

            pred_scores = pred_result.pred_scores.item.tolist()
            score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
            score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
            top5_label = score_sorted[:5]

            results = [(self.labels[k[0]], k[1]) for k in top5_label]
            kor_results = [(self.kor_labels[k[0]], k[1]) for k in top5_label]

            self.result_panel.append('<Top-5 labels with corresponding scores>')
            for result in results:
                self.result_panel.append(f'{result[0]}: {result[1]:.4f}')

            top1_kor_label = kor_results[0]
            if top1_kor_label[1] > 0.7:
                self.result_label.setText(f'{top1_kor_label[0]}')

        except Exception as e:
            print(f'Error occured: {e}')

    def process_webcam(self):
        self.score_cache = deque()
        self.scores_sum = 0
        print("Inference Thread Start...")
        self.inference_thread.start()

    def update_result(self, cur_time):
        cur_windows = []
        while len(cur_windows) == 0:
            if len(self.frame_queue) == self.sample_length:
                cur_windows = list(np.array(self.frame_queue))
                if self.data['img_shape'] is None:
                    self.data['img_shape'] = self.frame_queue.popleft().shape[:2]
        cur_data = self.data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = self.pipeline(cur_data)
        
        cur_data = pseudo_collate([cur_data])

        # Forward the model
        with torch.no_grad():
            result = self.model.test_step(cur_data)[0]
        scores = result.pred_scores.item.tolist()
        scores = np.array(scores)
        self.score_cache.append(scores)
        self.scores_sum += scores

        if len(self.score_cache) == self.average_size:
            scores_avg = self.scores_sum / self.average_size
            num_selected_labels = min(len(self.labels), 5)

            score_tuples = tuple(zip(self.labels, scores_avg))
            score_sorted = sorted(
                score_tuples, key=itemgetter(1), reverse=True)
            results = score_sorted[:num_selected_labels]

            self.result_queue.append(results)
            self.scores_sum -= self.score_cache.popleft()

            if len(self.result_queue) != 0 :
                self.result_panel.clear()
                self.result_label.setText("-")
                results = self.result_queue.popleft()
                self.result_panel.append('<Top-5 labels with corresponding scores>')
                
                top1_label = None
                for i, result in enumerate(results):
                    selected_label, score = result
                    if score < self.threshold:
                        break
                    if i == 0:
                        top1_label = result
                    self.result_panel.append(f'{selected_label}: {score:.4f}')

                if top1_label != None:
                    top1_kor_label = top1_label
                    kor_label = self.kor_labels[self.labels.index(top1_label[0])]
                    if top1_kor_label[1] > 0.7:
                        self.result_label.setText(f'{kor_label}')


    def closeEvent(self, event):
        self.webcam_thread.stop()
        self.inference_thread.stop()
        event.accept()
        

if __name__ == '__main__':
    args = parse_args()

    app = QApplication(sys.argv)
    fontDB = QFontDatabase()
    fontDB.addApplicationFont('demo/font/Supreme-Medium.otf')
    fontDB.addApplicationFont('demo/font/NotoSansKR-Bold.otf')
    app.setFont(QFont('Supreme Medium', 12))
    # app.setFont(QFont('Noto Sans KR', 13))
    # main_win = MainWindow()
    main_win = MainWindow(args)
    available_geometry = main_win.screen().availableGeometry()
    main_win.resize(available_geometry.width() / 3,
                    available_geometry.height() / 2)
    main_win.show()
    sys.exit(app.exec())
