import time
import numpy as np
from operator import itemgetter
from collections import deque

import torch
from PySide6.QtCore import QObject, QThread, Signal, Qt, Slot
from mmengine.dataset import Compose, pseudo_collate

class InferenceThread(QThread):
    infResult = Signal(list)

    def __init__(self):
        super().__init__()

    @Slot()
    def inference(self):
        self.infResult.emit(results)

