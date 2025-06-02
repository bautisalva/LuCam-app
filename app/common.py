import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import threading
import json
import datetime
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.io import imsave,imread
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, 
                             QSpinBox, QComboBox, QFileDialog,
                             QGroupBox, QTabWidget, QGridLayout,QPlainTextEdit)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QRect
from analisis_bordes import ImageEnhancer
from utils import blur_uint16

class SimulatedFrameFormat:
    def __init__(self):
        self.pixelFormat = None
        self.width = 640
        self.height = 480
        self.xOffset = 0
        self.yOffset = 0

class SimulatedCamera:
    """
    Fallback camera implementation used when Lucam is unavailable.
    Generates simulated grayscale images with added Gaussian noise and text overlay.

    Methods:
        - TakeSnapshot(): returns a noisy 480x640 grayscale image with overlaid text.
        - set_properties(**kwargs): stub to allow property configuration without functionality.
    """
    def TakeSnapshot(self):
        image = np.random.normal(32768, 5000, (480, 640)).astype(np.uint16)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        # podés usar una fuente default o cargar una con truetype
        draw.text((50, 240), "SIMULATED", fill=65535)
        image = np.array(img_pil)
        return image

    def set_properties(self, **kwargs):
        # Placeholder to mimic API compatibility with real Lucam camera.
        pass
    def GetFormat(self):
        return SimulatedFrameFormat(), 15.0  # Devuelve un formato dummy y 15 fps
    
    def SetFormat(self, frameformat, fps):
        # Ignorado en simulación
        pass
    
    def ContinuousAutoExposureDisable(self):
        # Ignorado en simulación
        pass
    
    def EnumAvailableFrameRates(self):
        return [7.5, 15.0, 30.0]  # Algunos valores típicos para simular


class Worker(QObject):
    """
    Worker class for image acquisition and processing in capture mode.
    
    Acquires multiple images from the camera and applies averaging or median filtering.
    Optionally applies Gaussian blur. Operates in a background thread.
    
    Signals:
        image_captured (np.ndarray): emitted with the final processed image.
    
    Parameters:
        camera (object): camera instance (Lucam or SimulatedCamera).
        num_images (int): number of images to acquire.
        mode (str): processing mode ('Promedio' or 'Mediana').
        blur_strength (int): Gaussian blur kernel half-width (0 disables blur).
    """
    image_captured = pyqtSignal(np.ndarray)

    def __init__(self, camera, num_images, mode, blur_strength):
        super().__init__()
        self.camera = camera
        self.num_images = num_images
        self.mode = mode
        self.blur_strength = blur_strength
        self._lock = threading.Lock()  # Add thread lock
        self._running = True  # Add running flag

    def run(self):
        try:
            images = []
            for _ in range(self.num_images):
                image = self.camera.TakeSnapshot()
                if image is None:
                    print("[WARNING] Imagen capturada fue None. Se salta.")
                    continue
                images.append(np.copy(image))

            if not images:
                print("[ERROR] No se capturó ninguna imagen válida.")
                return
            # Stack and process images based on the selected mode
            if self.mode == "Promedio":
                stack = np.stack(images).astype(np.float32)
                result_image = np.mean(stack, axis=0).astype(np.uint16).copy()
            elif self.mode == "Mediana":
                stack = np.stack(images).astype(np.uint16)
                result_image = np.median(stack, axis=0).astype(np.uint16).copy()
            else:
                print(f"[WARNING] Modo desconocido: {self.mode}, se usa la primera imagen.")
                result_image = images[0]
            # Optional Gaussian blur
            if self.blur_strength > 0:
                sigma = self.blur_strength  # o self.blur_strength / 3 para equivalente con OpenCV
                result_image = blur_uint16(result_image, sigma)
            # Emit final result
            self.image_captured.emit(result_image)

        except Exception:
            pass
            
    def stop(self):
        self._running = False
        
class PreviewWorker(QObject):
    """
    Worker class for acquiring and emitting live preview frames.

    Periodically acquires images from the camera and emits them as preview frames.
    Can be paused/resumed externally. Runs inside its own QThread.

    Signals:
        new_frame (np.ndarray): most recent preview image.

    Parameters:
        camera (object): camera instance used to acquire frames.
    """
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True
        self.paused = False

    def run(self):
        """
        Main loop: repeatedly captures frames and emits them unless paused.
        Uses QThread.msleep for frame pacing (300 ms).
        """
        while self.running:
            if not self.paused:
                image = self.camera.TakeSnapshot()
                if image is not None:
                    self.new_frame.emit(image)
            QThread.msleep(300)  # delay between frames

    def stop(self):
        """Stop the preview loop."""
        self.running = False

    def pause(self):
        """Temporarily suspend frame acquisition."""
        self.paused = True

    def resume(self):
        """Resume frame acquisition."""
        self.paused = False

class ROILabel(QLabel):
    roi_selected = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.start_pos = None
        self.end_pos = None
        self.drawing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.drawing = True
            self.end_pos = self.start_pos
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_pos = event.pos()
            self.update()

            x1 = min(self.start_pos.x(), self.end_pos.x())
            y1 = min(self.start_pos.y(), self.end_pos.y())
            x2 = max(self.start_pos.x(), self.end_pos.x())
            y2 = max(self.start_pos.y(), self.end_pos.y())

            width = x2 - x1
            height = y2 - y1

            self.roi_selected.emit(x1, y1, width, height)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing and self.start_pos and self.end_pos:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            rect = QRect(self.start_pos, self.end_pos)
            painter.drawRect(rect)

