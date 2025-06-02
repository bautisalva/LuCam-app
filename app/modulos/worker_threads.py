# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:10:32 2025

@author: Marina
"""

from PyQt5.QtCore import QObject, pyqtSignal, QThread
import numpy as np
import threading
from utils import blur_uint16


class Worker(QObject):
    image_captured = pyqtSignal(np.ndarray)

    def __init__(self, camera, num_images, mode, blur_strength):
        super().__init__()
        self.camera = camera
        self.num_images = num_images
        self.mode = mode
        self.blur_strength = blur_strength
        self._running = True

    def run(self):
        try:
            images = []
            for _ in range(self.num_images):
                image = self.camera.TakeSnapshot()
                if image is None:
                    continue
                images.append(np.copy(image))

            if not images:
                return

            if self.mode == "Promedio":
                stack = np.stack(images).astype(np.float32)
                result_image = np.mean(stack, axis=0).astype(np.uint16)
            elif self.mode == "Mediana":
                stack = np.stack(images).astype(np.uint16)
                result_image = np.median(stack, axis=0).astype(np.uint16)
            else:
                result_image = images[0]

            if self.blur_strength > 0:
                result_image = blur_uint16(result_image, self.blur_strength)

            self.image_captured.emit(result_image)
        except Exception as e:
            print(f"[ERROR] Worker thread: {e}")

    def stop(self):
        self._running = False


class PreviewWorker(QObject):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True
        self.paused = False

    def run(self):
        while self.running:
            if not self.paused:
                image = self.camera.TakeSnapshot()
                if image is not None:
                    self.new_frame.emit(image)
            QThread.msleep(300)

    def stop(self):
        self.running = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
