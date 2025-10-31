"""
LuCam Camera Control Application

This module provides a PyQt6-based graphical interface for controlling a Lumenera camera (via the `lucam` module),
or a simulated fallback camera when the real hardware is unavailable. It offers live preview, background subtraction,
image acquisition with averaging/median and optional Gaussian blur, adjustable capture parameters, and export to disk.

Classes:
    - SimulatedCamera: Fallback for Lucam hardware.
    - Worker: Handles background-threaded image capture and processing.
    - PreviewWorker: Manages real-time frame preview.
    - CameraApp: Main application window and GUI logic.


Author: Tomás Rodriguez Bouhier, Bautista Salvatierra Pérez
Repository: https://github.com/bautisalva/LuCam-app
mirko pedazo de gato
bbb
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import threading
import json
import datetime
import time
import pyvisa as visa
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.io import imsave
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, 
                             QSpinBox, QComboBox, QFileDialog, QMessageBox,
                             QGroupBox, QTabWidget, QGridLayout, QPlainTextEdit,
                             QInputDialog, QCheckBox, QButtonGroup, QRadioButton,
                             QMainWindow, QTableWidgetItem,QHeaderView, QProgressBar,
                             QTableWidget,QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QSpinBox, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen,QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QRect
from analysis_tab import AnalysisTab


def blur_uint16(image, sigma):
    """
    Applies Gaussian blur to a 16-bit image using scipy.ndimage.
    
    Parameters:
        image (np.ndarray): uint16 image.
        sigma (float): standard deviation of Gaussian kernel.
    
    Returns:
        np.ndarray: blurred image, dtype uint16.
    """
    return gaussian_filter(image, sigma=sigma, mode='reflect').astype(np.uint16)

def to_8bit_for_preview(image_16bit):
    """
    Escala una imagen uint16 a uint8 para visualización,
    mapeando el rango [min, max] a [0, 255].
    """
    min_val = np.min(image_16bit)
    max_val = np.max(image_16bit)
    if max_val == min_val:
        return np.zeros_like(image_16bit, dtype=np.uint8)
    scaled = ((image_16bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return scaled


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


# Try to import Lucam
try:
    from lucam import Lucam
    from lucam import API
    LUCAM_AVAILABLE = True
except ImportError:
    LUCAM_AVAILABLE = False
    print("[WARNING] Módulo 'lucam' no disponible, usando modo simulación.")

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
            QThread.msleep(100)  # delay between frames

    def stop(self):
        """Stop the preview loop."""
        self.running = False

    def pause(self):
        """Temporarily suspend frame acquisition."""
        self.paused = True

    def resume(self):
        """Resume frame acquisition."""
        self.paused = False


class SequenceWorker(QObject):
    progress = pyqtSignal(int, str)
    bg_ready = pyqtSignal(np.ndarray)
    image_ready = pyqtSignal(np.ndarray, int)  # (img_roi_16bit, idx)
    finished = pyqtSignal()

    # Señales para que la APP ejecute los pulsos usando LO QUE YA ESTÁ SELECCIONADO en la UI
    do_saturate = pyqtSignal()
    do_nucleation = pyqtSignal()
    do_growth = pyqtSignal()

    def __init__(self, *, camera, cycles, roi, do_resta,
                 background_gain, background_offset,
                 num_images_bg=1, num_images_frame=1,
                 blur_sigma=0, mode="Promedio",
                 outdir="", parent_log=None):
        super().__init__()
        self.camera = camera
        self.cycles = int(cycles)
        self.roi = roi  # (x,y,w,h) o None
        self.do_resta = bool(do_resta)
        self.bg_gain = float(background_gain)
        self.bg_offset = float(background_offset)
        self.num_images_bg = int(num_images_bg)
        self.num_images_frame = int(num_images_frame)
        self.blur_sigma = int(blur_sigma)
        self.mode = mode or "Promedio"
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(os.path.join(self.outdir, "raw"), exist_ok=True)
        self.parent_log = parent_log or (lambda m: None)
        self._stop = False
        self.background = None

    # ---------- helpers ----------
    def log(self, m):
        self.parent_log(m)

    def stop(self):
        self._stop = True

    def should_stop(self):
        return self._stop

    def _stack_capture(self, n):
        imgs = []
        for _ in range(max(1, n)):
            im = self.camera.TakeSnapshot()
            if im is not None:
                imgs.append(np.copy(im))
        if not imgs:
            raise RuntimeError("No se pudo capturar ninguna imagen.")
        if len(imgs) == 1:
            out = imgs[0]
        else:
            stack = np.stack(imgs, 0).astype(np.float32)
            mode_lc = str(self.mode).lower()
            if mode_lc.startswith("med"):
                out = np.median(stack, axis=0).astype(np.uint16)
            else:
                out = np.mean(stack, axis=0).astype(np.uint16)
        if self.blur_sigma > 0:
            out = blur_uint16(out, sigma=self.blur_sigma)
        return out

    def _apply_roi(self, img):
        if self.roi is None:
            h, w = img.shape
            return img, (0, 0, w, h)
        x, y, w, h = self.roi
        x = max(0, x); y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        return img[y:y+h, x:x+w], (x, y, w, h)

    def _apply_resta(self, img_roi, roi_box):
        if (self.background is None) or (not self.do_resta):
            return img_roi
        x, y, w, h = roi_box
        bg_roi = self.background[y:y+h, x:x+w]
        a = self.bg_gain; b = self.bg_offset
        diff = a * (img_roi.astype(np.float32) - bg_roi.astype(np.float32) + b)
        mn, mx = float(diff.min()), float(diff.max())
        if mx <= mn:
            return np.zeros_like(img_roi, dtype=np.uint16)
        return ((diff - mn) / (mx - mn) * 65535).astype(np.uint16)

    def _save_triplet(self, full, roi_img, roi_box, stem):
        from skimage.io import imsave as _imsave
        _imsave(os.path.join(self.outdir, "raw", f"{stem}_cruda.tif"), full)
        _imsave(os.path.join(self.outdir, f"{stem}_normal.tif"), roi_img)
        if self.do_resta and (self.background is not None):
            resta = self._apply_resta(roi_img, roi_box)
            _imsave(os.path.join(self.outdir, f"{stem}_resta.tif"), resta)

    # ---------- main ----------
    def run(self):
        try:
            # 0) Saturar una vez con la UI actual
            self.log("[SEQ] Saturando (según selección actual)...")
            self.do_saturate.emit()
            time.sleep(0.05)
            self.progress.emit(5, "Saturación OK")
            if self.should_stop():
                self.finished.emit(); return

            # 1) Fondo una sola vez
            self.log("[SEQ] Capturando fondo...")
            bg = self._stack_capture(self.num_images_bg)
            self.background = bg.copy()
            self.bg_ready.emit(bg.copy())
            from skimage.io import imsave as _imsave
            _imsave(os.path.join(self.outdir, "fondo.tif"), bg)
            self.progress.emit(20, "Fondo capturado")
            if self.should_stop():
                self.finished.emit(); return

            # 2) Nucleación y captura inicial antes de crecimiento
            if self.should_stop():
                self.finished.emit(); return

            self.log("[SEQ] Enviando nucleación inicial...")
            self.do_nucleation.emit()
            time.sleep(0.01)

            nuc_full = self._stack_capture(self.num_images_frame)
            nuc_roi, nuc_box = self._apply_roi(nuc_full)
            self._save_triplet(nuc_full, nuc_roi, nuc_box, "nucleacion")
            self.image_ready.emit(nuc_roi.copy(), 0)
            self.progress.emit(30, "Nucleación capturada")
            if self.should_stop():
                self.finished.emit(); return

            # 3) Ciclos de crecimiento = fotos
            total = max(0, self.cycles)
            for k in range(1, total + 1):
                if self.should_stop():
                    break

                # Paso de crecimiento (usa 'características del ciclo' actuales)
                self.log(f"[SEQ] Ciclo {k}/{total}: crecimiento")
                self.do_growth.emit()
                time.sleep(0.01)

                full = self._stack_capture(self.num_images_frame)
                roi_img, roi_box = self._apply_roi(full)
                stem = f"frame_{k:04d}"
                self._save_triplet(full, roi_img, roi_box, stem)

                self.image_ready.emit(roi_img.copy(), k)

                pct = 30 + int(70 * k / max(1, total))
                self.progress.emit(pct, f"Ciclo {k}/{total} listo")

            if total == 0:
                self.progress.emit(100, "Secuencia completada")

        except Exception as e:
            self.log(f"[ERROR][SEQ] {e}")
        finally:
            self.finished.emit()


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


class CameraApp(QWidget):
    """
    Main GUI application class for the LuCam interface.

    Encapsulates GUI setup, user interaction logic, camera property control,
    image previewing, background subtraction, and acquisition workflows.

    Uses:
        - Lucam camera (if available)
        - SimulatedCamera (fallback)
        - PyQt6 for GUI and threading
    """
    seq_log_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
    
        # Try initializing Lucam camera; fallback to simulation
        try:
            self.camera = Lucam()
            self.camera.CameraClose()  # Cerrá por si quedó colgada
            self.camera = Lucam()      # Volvé a abrirla limpia
            self.simulation = False
            print("[INFO] Cámara Lucam reiniciada al iniciar.")

        except Exception as e:
            print(f"[WARNING] No se pudo inicializar Lucam. Se usará SimulatedCamera. Error: {e}")
            self.camera = SimulatedCamera()
            self.simulation = True

        self.roi_enabled = False
        self.last_full_image = None  # Guarda imagen sin ROI

        # Configuración del formato del frame
        frameformat, fps = self.camera.GetFormat()
        frameformat.pixelFormat = API.LUCAM_PF_16
        self.camera.SetFormat(frameformat, fps)
        
        self.frame_width = frameformat.width
        self.frame_height = frameformat.height
    
        # Logging system
        self.log_file_path = os.path.join(os.getcwd(), "log.txt")
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
        
        now = datetime.datetime.now()
        start_message = f"=== Se inició la app el día {now.strftime('%d/%m/%Y')} a las {now.strftime('%H:%M:%S')} ==="
        self.log_file.write(start_message + "\n")
        self.log_file.flush()

        # Redirigir logs generados desde hilos secundarios al hilo principal
        self.seq_log_signal.connect(self.log_message)

        # Start preview worker thread
        self.preview_worker = PreviewWorker(self.camera)
        self.preview_thread = QThread()
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_worker.new_frame.connect(self.display_preview_image)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_thread.start()
    
        # Initialize GUI widgets and internal state variables
        self.preview_label_preview = QLabel("Preview en vivo")
        self.preview_label_preview.setFixedSize(960, 786)
    
        self.preview_label_capture = ROILabel()
        self.preview_label_capture.roi_selected.connect(self.set_roi_from_mouse)
        self.preview_label_capture.setFixedSize(960, 786)
    
        self.console_preview = QPlainTextEdit()
        self.console_preview.setReadOnly(True)
    
        self.console_capture = QPlainTextEdit()
        self.console_capture.setReadOnly(True)
    
        # Internal states for background subtraction and UI control
        self.background_gain = 1.0
        self.background_offset = 0.0
        self.preview_mode = True
        self.capture_mode = "Promedio"
        self.captured_image = None
        self.blur_strength = 0
        self.background_image = None
        self.background_enabled = True
    
        # Properties to be controlled by UI sliders
        self.properties = {
            "brightness": (0, 100, 10.0),
            "contrast": (0, 10, 1.0),
            "saturation": (0, 100, 10.0),
            "hue": (-180, 180, 0.0),
            "gamma": (1, 5, 10),
            "exposure": (1, 375, 10.0),
            "gain": (0, 7.75, 1.0),
        }
    
        #self.camera.SetProperty(168,10)
        self.camera.ContinuousAutoExposureDisable()
        
        try:
            self.available_fps = self.camera.EnumAvailableFrameRates()
            self.available_fps = [round(f, 2) for f in self.available_fps]
        except Exception as e:
            self.available_fps = [7.5, 15.0]  # valor por defecto
            self.log_message(f"[WARNING] No se pudieron obtener los FPS disponibles: {e}")
    
        #self.set_roi(1280, 1048)
    
        self.work_dir = ""
        self.auto_save = False
        self.setWindowIcon(QIcon("toto.png"))
        

        # Launch full GUI setup
        self.initUI()



    def initUI(self):
        """
        Constructs the main interface layout of the application.
        Creates tab widget with 'Preview' and 'Captura' tabs,
        each containing its respective layout and controls.
        """
        
        self.setWindowTitle("Lumenera Camera Control")
        self.setGeometry(100, 100, 1000, 600)
    
        # Creates windows
        self.tabs = QTabWidget(self)
    
        # Window 1: Preview
        self.preview_tab = QWidget()
        self.init_preview_tab()
    
        # Window 2: Capture
        self.capture_tab = QWidget()
        self.init_capture_tab()
        
        # Window 3: Analysis
        self.analysis_tab = AnalysisTab(self.get_last_image, self.log_message)
        
        # Window 4: Pulse control
        self.control_tab = QWidget()
        self.init_control_tab()
    
        # Add window
        self.tabs.addTab(self.preview_tab, "Preview")
        self.tabs.addTab(self.capture_tab, "Captura")
        self.tabs.addTab(self.analysis_tab, "Análisis de Imagen")
        self.tabs.addTab(self.control_tab, "Control de pulso")

    
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
        
    def init_preview_tab(self):
        """
        Initializes widgets and layout for the preview tab.
        Includes live preview display, camera property sliders,
        and options to save/load preview parameters.
        """
        layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.preview_label_preview)
        left_layout.addWidget(self.console_preview)
    
        layout.addLayout(left_layout)
    
        controls_layout = QVBoxLayout()
        self.sliders = {}
        self.inputs = {}
    
        fps_group = QGroupBox("FPS")
        fps_layout = QHBoxLayout()
        self.fps_selector = QComboBox()
        self.fps_selector.setStyleSheet("background-color: lightyellow;")  # opcional para que lo veas
        for fps in self.available_fps:
            self.fps_selector.addItem(f"{fps:.2f}")
        self.fps_selector.currentTextChanged.connect(self.change_fps)
        fps_layout.addWidget(QLabel("Frames por segundo:"))
        fps_layout.addWidget(self.fps_selector)
        fps_group.setLayout(fps_layout)
        controls_layout.addWidget(fps_group)
    
        try:
            current_fps = self.camera.GetFormat()[1]
            index = self.fps_selector.findText(f"{current_fps:.2f}")
            if index != -1:
                self.fps_selector.setCurrentIndex(index)
        except Exception as e:
            self.log_message(f"[WARNING] No se pudo establecer FPS actual: {e}")
    
        for prop, (min_val, max_val, default) in self.properties.items():
            group = QGroupBox(prop.capitalize())
            group_layout = QHBoxLayout()
    
            label = QLabel(f"{default}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(min_val * 100))
            slider.setMaximum(int(max_val * 100))
            slider.setValue(int(default * 100))
            slider.valueChanged.connect(lambda value, p=prop: self.update_property(p, value / 100))
    
            input_field = QLineEdit(str(default))
            input_field.setFixedWidth(50)
            input_field.editingFinished.connect(lambda p=prop, field=input_field: self.set_property_from_input(p, field))
    
            group_layout.addWidget(label)
            group_layout.addWidget(slider)
            group_layout.addWidget(input_field)
            group.setLayout(group_layout)
    
            controls_layout.addWidget(group)
            self.sliders[prop] = slider
            self.inputs[prop] = input_field
    
        # Save/load parameter buttons
        self.save_preview_button = QPushButton("Guardar Parámetros de Preview")
        self.save_preview_button.clicked.connect(self.save_preview_parameters)
        controls_layout.addWidget(self.save_preview_button)
    
        self.load_settings_button = QPushButton("Cargar Parámetros")
        self.load_settings_button.clicked.connect(self.load_parameters)
        controls_layout.addWidget(self.load_settings_button)
        
        self.refresh_preview_button = QPushButton("Refrescar desde Cámara")
        self.refresh_preview_button.clicked.connect(self.apply_real_camera_values_to_sliders)
        controls_layout.addWidget(self.refresh_preview_button)
    
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
    
        self.apply_real_camera_values_to_sliders()
    
        self.preview_tab.setLayout(layout)

    def init_capture_tab(self):
        """
        Initializes layout and widgets for the image capture tab.
        Includes settings for capture mode, background subtraction,
        saving, Gaussian blur, and acquisition logic.
        """
        layout = QHBoxLayout()
    
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.preview_label_capture)
        left_layout.addWidget(self.console_capture)
        
        layout.addLayout(left_layout)

        controls_layout = QVBoxLayout()
    
        # Directory selector
        dir_layout = QHBoxLayout()
        self.dir_line_edit = QLineEdit()
        self.dir_button = QPushButton("...")
        self.dir_button.setFixedWidth(30)
        self.dir_button.clicked.connect(self.select_work_dir)
        dir_layout.addWidget(QLabel("Directorio:"))
        dir_layout.addWidget(self.dir_line_edit)
        dir_layout.addWidget(self.dir_button)
        controls_layout.addLayout(dir_layout)
    
        # Auto-save toggle
        auto_save_layout = QHBoxLayout()
        self.auto_save_selector = QComboBox()
        self.auto_save_selector.addItems(["No", "Sí"])
        self.auto_save_selector.currentTextChanged.connect(self.toggle_auto_save)
        auto_save_layout.addWidget(QLabel("Guardar automáticamente:"))
        auto_save_layout.addWidget(self.auto_save_selector)
        controls_layout.addLayout(auto_save_layout)
    
        # Capture settings
        capture_settings_box = QGroupBox("Configuraciones de Captura")
        capture_settings_layout = QVBoxLayout()
    
        self.num_images_spinbox = QSpinBox()
        self.num_images_spinbox.setRange(1, 100)
        self.num_images_spinbox.setValue(5)
        capture_num_layout = QHBoxLayout()
        capture_num_layout.addWidget(QLabel("Imágenes por captura:"))
        capture_num_layout.addWidget(self.num_images_spinbox)
    
        self.capture_mode_selector = QComboBox()
        self.capture_mode_selector.addItems(["Promedio", "Mediana"])
        self.capture_mode_selector.currentTextChanged.connect(self.change_capture_mode)
        capture_mode_layout = QHBoxLayout()
        capture_mode_layout.addWidget(QLabel("Modo de captura:"))
        capture_mode_layout.addWidget(self.capture_mode_selector)
    
        # Blur controls
        blur_box = QGroupBox("Desenfoque")
        blur_layout = QHBoxLayout()
        self.blur_label = QLabel("Blur: 0")
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setMinimum(0)
        self.blur_slider.setMaximum(10)
        self.blur_slider.setValue(0)
        self.blur_slider.valueChanged.connect(self.update_blur)
        self.blur_input = QLineEdit("0")
        self.blur_input.setFixedWidth(50)
        self.blur_input.editingFinished.connect(self.set_blur_from_input)
    
        blur_layout.addWidget(self.blur_label)
        blur_layout.addWidget(self.blur_slider)
        blur_layout.addWidget(self.blur_input)
        blur_box.setLayout(blur_layout)
    
        capture_settings_layout.addLayout(capture_num_layout)
        capture_settings_layout.addLayout(capture_mode_layout)
        capture_settings_layout.addWidget(blur_box)
        capture_settings_box.setLayout(capture_settings_layout)
        controls_layout.addWidget(capture_settings_box)
        
        # Background subtraction settings
        gain_offset_box = QGroupBox("Fondo: Ganancia y Offset")
        gain_offset_layout = QVBoxLayout()

        self.capture_button = QPushButton("Capturar Imagen")
        self.capture_button.clicked.connect(self.capture_image)
        controls_layout.addWidget(self.capture_button)
    
        self.capture_background_button = QPushButton("Capturar Fondo")
        self.capture_background_button.clicked.connect(self.capture_background)
        controls_layout.addWidget(self.capture_background_button)
    
        toggle_background_layout = QHBoxLayout()
        toggle_background_label = QLabel("Restar fondo:")
        self.toggle_background_selector = QComboBox()
        self.toggle_background_selector.addItems(["Sí", "No"])
        self.toggle_background_selector.currentTextChanged.connect(self.toggle_background)
        toggle_background_layout.addWidget(toggle_background_label)
        toggle_background_layout.addWidget(self.toggle_background_selector)
        controls_layout.addLayout(toggle_background_layout)
    
        # Gain
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Ganancia (a):"))
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setMinimum(0)
        self.gain_slider.setMaximum(10000)
        self.gain_slider.setValue(100)
        self.gain_slider.valueChanged.connect(lambda v: self.update_gain(v / 100))
        gain_layout.addWidget(self.gain_slider)
        self.gain_input = QLineEdit("1.0")
        self.gain_input.setFixedWidth(50)
        self.gain_input.editingFinished.connect(lambda: self.set_gain_from_input())
        gain_layout.addWidget(self.gain_input)

        # Offset
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Offset (b):"))
        self.offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.offset_slider.setMinimum(-100)
        self.offset_slider.setMaximum(100)
        self.offset_slider.setValue(0)
        self.offset_slider.valueChanged.connect(lambda v: self.update_offset(v / 100))
        offset_layout.addWidget(self.offset_slider)
        self.offset_input = QLineEdit("0")
        self.offset_input.setFixedWidth(50)
        self.offset_input.editingFinished.connect(lambda: self.set_offset_from_input())
        offset_layout.addWidget(self.offset_input)

        gain_offset_layout.addLayout(gain_layout)
        gain_offset_layout.addLayout(offset_layout)
        gain_offset_box.setLayout(gain_offset_layout)
        controls_layout.addWidget(gain_offset_box)
        
        self.save_button = QPushButton("Guardar Imagen")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        controls_layout.addWidget(self.save_button)
    
        self.toggle_preview_button = QPushButton("Volver a Preview")
        self.toggle_preview_button.clicked.connect(self.start_preview)
        self.toggle_preview_button.setEnabled(False)
        controls_layout.addWidget(self.toggle_preview_button)
    
        self.save_capture_button = QPushButton("Guardar Parámetros de Captura")
        self.save_capture_button.clicked.connect(self.save_capture_parameters)
        controls_layout.addWidget(self.save_capture_button)

    
        self.load_settings_button = QPushButton("Cargar Parámetros")
        self.load_settings_button.clicked.connect(self.load_parameters)
        controls_layout.addWidget(self.load_settings_button)
        
        # ROI manual
        roi_box = QGroupBox("Región de Interés (ROI)")
        roi_layout = QGridLayout()
        
        self.roi_x_input = QSpinBox()
        self.roi_x_input.setRange(0, self.frame_width)
        self.roi_y_input = QSpinBox()
        self.roi_y_input.setRange(0, self.frame_height)
        self.roi_width_input = QSpinBox()
        self.roi_width_input.setRange(8, self.frame_width)
        self.roi_height_input = QSpinBox()
        self.roi_height_input.setRange(8, self.frame_height)
        
        roi_layout.addWidget(QLabel("X:"), 0, 0)
        roi_layout.addWidget(self.roi_x_input, 0, 1)
        roi_layout.addWidget(QLabel("Y:"), 0, 2)
        roi_layout.addWidget(self.roi_y_input, 0, 3)
        roi_layout.addWidget(QLabel("Ancho:"), 1, 0)
        roi_layout.addWidget(self.roi_width_input, 1, 1)
        roi_layout.addWidget(QLabel("Alto:"), 1, 2)
        roi_layout.addWidget(self.roi_height_input, 1, 3)

        roi_box.setLayout(roi_layout)
        controls_layout.addWidget(roi_box)

        # Toggle ROI
        toggle_roi_layout = QHBoxLayout()
        toggle_roi_label = QLabel("Aplicar ROI:")
        self.toggle_roi_selector = QComboBox()
        self.toggle_roi_selector.addItems(["Sí", "No"])
        self.toggle_roi_selector.setCurrentIndex(1)  # "No"

        self.toggle_roi_selector.currentTextChanged.connect(self.toggle_roi)
        toggle_roi_layout.addWidget(toggle_roi_label)
        toggle_roi_layout.addWidget(self.toggle_roi_selector)
        controls_layout.addLayout(toggle_roi_layout)

        controls_layout.addStretch()
    
        layout.addLayout(controls_layout)
    
        self.capture_tab.setLayout(layout)
    
    def init_control_tab(self):
        '''
        This tab has the porpuse of controling de intensity and width of the magnetic pulses that modify the PDM's
        '''
        main_layout = QVBoxLayout()

         # === BOTÓN DE INICIO ===
        self.init_button = QPushButton("INICIAR")
        self.init_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.init_button.clicked.connect(self.iniciar_conexion)
        main_layout.addWidget(self.init_button)

        # --- Main frame ---
        main_group = QGroupBox("Control de pulsos y toma de datos")
        main_group_layout = QVBoxLayout(main_group)

        # --- Osciloscope ---
        osc_group = QGroupBox("Osciloscopio")
        osc_layout = QGridLayout(osc_group)

        osc_layout.addWidget(QLabel("Nombre archivo datos:"), 0, 0)
        self.osc_file_edit = QLineEdit()
        osc_layout.addWidget(self.osc_file_edit, 0, 1,1,3)

        osc_layout.addWidget(QLabel("Escala X:"), 1, 2)
        self.osc_x_combo = QComboBox()
        self.osc_x_combo.addItems(["1 ns/div", "2 ns/div", "5 ns/div", 
                                  "10 ns/div", "20 ns/div", "50 ns/div", 
                                  "100 ns/div", "200 ns/div", "500 ns/div", 
                                  "1 µs/div", "2 µs/div", "5 µs/div", 
                                  "10 µs/div", "20 µs/div", "50 µs/div", 
                                  "100 µs/div", "200 µs/div", "500 µs/div", 
                                  "1 ms/div", "2 ms/div", "5 ms/div", 
                                  "10 ms/div", "20 ms/div", "50 ms/div", 
                                  "100 ms/div", "200 ms/div", "500 ms/div", 
                                  "1 s/div", "2 s/div", "5 s/div", "10 s/div"])  
        osc_layout.addWidget(self.osc_x_combo,2,2)

        osc_layout.addWidget(QLabel("Escala Y:"), 1, 3)
        self.osc_y_combo = QComboBox()
        self.osc_y_combo.addItems(["1 mV/div", "2 mV/div", "5 mV/div", 
                                   "10 mV/div", "20 mV/div", "50 mV/div", 
                                   "100 mV/div", "200 mV/div", "500 mV/div", 
                                   "1 V/div", "2 V/div", "5 V/div", "10 V/div"])  
        osc_layout.addWidget(self.osc_y_combo, 2, 3)

        self.osci_scale_button = QPushButton("Cambiar Escala")
        osc_layout.addWidget(self.osci_scale_button,3,2,1,2)

        self.osc_save_checkbox = QCheckBox("Guardar foto del osciloscopio")
        osc_layout.addWidget(self.osc_save_checkbox, 1,0, 1, 2)

        self.osci_data_reader_button = QPushButton("Tomar datos del Osciloscopio")
        osc_layout.addWidget(self.osci_data_reader_button,4,0,1,4)

        main_group_layout.addWidget(osc_group)

        # === GUARDAMOS referencias para deshabilitarlas inicialmente ===
        self.osc_buttons = [self.osci_scale_button, self.osci_data_reader_button]
        for btn in self.osc_buttons:
            btn.setEnabled(False)

        # --- Mid Frame ---
        mid_layout = QHBoxLayout()

        dom_group = QGroupBox("Saturar muestra y crear dominios")
        dom_layout = QGridLayout(dom_group)

        dom_layout.addWidget(QLabel("Tiempo de saturación [ms]"), 0, 0)
        self.tiempo_saturacion_edit = QLineEdit()
        dom_layout.addWidget(self.tiempo_saturacion_edit, 0, 1)

        dom_layout.addWidget(QLabel("Campo de saturación [Oe]"), 1, 0)
        self.campo_saturacion_edit = QLineEdit()
        dom_layout.addWidget(self.campo_saturacion_edit, 1, 1)

        dom_layout.addWidget(QLabel("Tiempo de dominio [ms]"), 0, 2)
        self.tiempo_dominio_edit = QLineEdit()
        dom_layout.addWidget(self.tiempo_dominio_edit, 0, 3)

        dom_layout.addWidget(QLabel("Campo de dominio [Oe]"), 1, 2)
        self.campo_dominio_edit = QLineEdit()
        dom_layout.addWidget(self.campo_dominio_edit, 1, 3)

        # dom_layout.addWidget(QLabel("Tipo de pulso"),2,1)
        # self.combo_pulso = QComboBox()
        # self.combo_pulso.addItems(["Pulso Pos.+","Pulso Neg.-","Pulso Mixto", "Pulso Oscilatorio"])
        # dom_layout.addWidget(self.combo_pulso,2,2)

        dom_layout.addWidget(QLabel("Tipo de pulso:"),2,0)
        # self.combo_pulso = QComboBox()
        # self.combo_pulso.addItems(["Pulso Pos.+","Pulso Neg.-","Pulso Mixto", "Pulso Oscilatorio"])
        # dom_layout.addWidget(self.combo_pulso,2,1)
        dom_layout.addWidget(QLabel("Cuadrado"),2,1)

        dom_layout.addWidget(QLabel("Signo del dominio:"), 2, 2)

        self.radio_signo_pos_dom = QRadioButton("Positivo")
        self.radio_signo_neg_dom = QRadioButton("Negativo")

        # Para que sean excluyentes, van en un mismo QButtonGroup
        self.signo_group_dom = QButtonGroup()
        self.signo_group_dom.addButton(self.radio_signo_pos_dom)
        self.signo_group_dom.addButton(self.radio_signo_neg_dom)

        signo_layout_dom = QHBoxLayout()
        signo_layout_dom.addWidget(self.radio_signo_pos_dom)
        signo_layout_dom.addWidget(self.radio_signo_neg_dom)

        dom_layout.addLayout(signo_layout_dom, 2, 3)

        self.saturate_dom_button = QPushButton("Saturar")
        self.create_dom_button = QPushButton("Crear dominios")
        dom_layout.addWidget(self.saturate_dom_button, 3, 1)
        dom_layout.addWidget(self.create_dom_button, 3, 2)

        mid_layout.addWidget(dom_group)

        main_group_layout.addLayout(mid_layout)

        # --- Combobox for configurations already saved ---
        combo_group = QGroupBox()
        combo_layout = QGridLayout(combo_group)

        combo_layout.addWidget(QLabel("Seleccionar configuración"), 0, 0) 
        self.combo = QComboBox() 
        try: 
          with open("../../params/params_preconfiguration.json", "r", encoding="utf-8") as f: 
            #Everything will be saved in here 
            data = json.load(f) 
            for nombre, valores in data.items():   # nombre = clave, valores = diccionario
                self.combo.addItem(nombre, valores)
        except Exception as e: 
            print(f"[WARNING] No se pudo cargar JSON: {e}")
            
        combo_layout.addWidget(self.combo, 0, 1)

        main_group_layout.addWidget(combo_group)


        # --- Inferior Frame ---
        bottom_layout = QHBoxLayout()

        # --- Cicle caracteristics ---
        
        ciclo_group = QGroupBox("Características Ciclo")
        ciclo_layout = QGridLayout(ciclo_group)

        ciclo_layout.addWidget(QLabel("Constate campo-corriente [G/mA]:"), 0, 0)
        self.campo_corr_edit = QLineEdit()
        ciclo_layout.addWidget(self.campo_corr_edit, 0, 1)

        ciclo_layout.addWidget(QLabel("Resistencia [Ω]:"), 0, 2)
        self.resistencia_edit = QLineEdit()
        ciclo_layout.addWidget(self.resistencia_edit, 0, 3)

        ciclo_layout.addWidget(QLabel("Campo [Oe]:"), 1, 0)
        self.campo_ciclo_edit = QLineEdit()
        ciclo_layout.addWidget(self.campo_ciclo_edit, 1, 1)

        ciclo_layout.addWidget(QLabel("Tiempo [ms]:"), 1, 2)
        self.campo_ciclo_edit = QLineEdit()
        ciclo_layout.addWidget(self.campo_ciclo_edit, 1, 3)

        ciclo_layout.addWidget(QLabel("Offset [V]:"), 2, 0)
        self.offset_ciclo_edit = QLineEdit()
        ciclo_layout.addWidget(self.offset_ciclo_edit, 2, 1)

        ciclo_layout.addWidget(QLabel("Nro de ciclos:"), 2, 2)
        self.nro_ciclo_edit = QLineEdit()
        ciclo_layout.addWidget(self.nro_ciclo_edit, 2, 3)

        ciclo_layout.addWidget(QLabel("Tipo de pulso:"),3,0)
        self.combo_pulso_ciclo = QComboBox()
        self.combo_pulso_ciclo.addItems(["Pulso cuadrado","Pulso Mixto", "Pulso Oscilatorio"])
        ciclo_layout.addWidget(self.combo_pulso_ciclo,3,1)

        # --- Campos adicionales (ocultos inicialmente) ---
        # Oscilatorio
        self.label_amp = QLabel("Amplitud de oscilación [%] (0-50):")
        self.amp_edit = QLineEdit()
        self.label_frec = QLabel("Cantidad de oscilaciones:")
        self.frec_edit = QLineEdit()

        ciclo_layout.addWidget(self.label_amp, 4, 0)
        ciclo_layout.addWidget(self.amp_edit, 4, 1)
        ciclo_layout.addWidget(self.label_frec, 4, 2)
        ciclo_layout.addWidget(self.frec_edit, 4, 3)

        # Asimétrico
        self.label_asim = QLabel("% de asimetría:")
        self.asim_edit = QLineEdit()
        self.label_altura = QLabel("Altura de asimetría [V]:")
        self.altura_edit = QLineEdit()

        ciclo_layout.addWidget(self.label_asim, 5, 0)
        ciclo_layout.addWidget(self.asim_edit, 5, 1)
        ciclo_layout.addWidget(self.label_altura, 5, 2)
        ciclo_layout.addWidget(self.altura_edit, 5, 3)

        # Ocultamos todos los campos adicionales al inicio
        for widget in [self.label_amp, self.amp_edit, self.label_frec, self.frec_edit,
                    self.label_asim, self.asim_edit, self.label_altura, self.altura_edit]:
            widget.hide()

        # Función de actualización
        def actualizar_campos_pulso(tipo):
            # Ocultamos todos
            for widget in [self.label_amp, self.amp_edit, self.label_frec, self.frec_edit,
                        self.label_asim, self.asim_edit, self.label_altura, self.altura_edit]:
                widget.hide()

            if tipo == "Pulso Oscilatorio":
                self.label_amp.show()
                self.amp_edit.show()
                self.label_frec.show()
                self.frec_edit.show()

            elif tipo == "Pulso Mixto":
                self.label_asim.show()
                self.asim_edit.show()
                self.label_altura.show()
                self.altura_edit.show()

        # Conexión del combo a la función
        self.combo_pulso_ciclo.currentTextChanged.connect(actualizar_campos_pulso)


        bottom_layout.addWidget(ciclo_group)
        
        self.gen_buttons=[self.saturate_dom_button,self.create_dom_button]
        for btn in self.gen_buttons:
            btn.setEnabled(False)

        # --- Saturate and create domains ---

        capture_group = QGroupBox("Captura")
        capture_layout = QGridLayout(capture_group)
        
        bottom_layout.addWidget(capture_group)

        def _add_sequence_block_to_control_tab(self, parent_layout):
            seq_group = QGroupBox("Secuencia (usa selecciones actuales de Saturación/Nucleación y 'Características del ciclo')")
            left = QVBoxLayout(seq_group)
        
            row1 = QHBoxLayout()
            row1.addWidget(QLabel("Ciclos (fotos totales):"))
            self.seq_cycles = QSpinBox(); self.seq_cycles.setRange(1, 10000); self.seq_cycles.setValue(10)
            row1.addWidget(self.seq_cycles)
            left.addLayout(row1)
        
            row2 = QHBoxLayout()
            self.seq_run_btn = QPushButton("Ejecutar secuencia")
            self.seq_stop_btn = QPushButton("Detener")
            self.seq_stop_btn.setEnabled(False)
            self.seq_run_btn.clicked.connect(self._seq_run)
            self.seq_stop_btn.clicked.connect(self._seq_stop)
            row2.addWidget(self.seq_run_btn); row2.addWidget(self.seq_stop_btn)
            left.addLayout(row2)
        
            self.seq_progress = QLabel("Listo")
            left.addWidget(self.seq_progress)
        
            # Vista previa a la derecha
            preview_group = QGroupBox("Última foto")
            pr_layout = QVBoxLayout(preview_group)
            self.seq_preview = QLabel("\n\n(Se mostrará aquí)")
            self.seq_preview.setAlignment(Qt.AlignCenter)
            self.seq_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.seq_preview.setMinimumSize(260, 200)
            pr_layout.addWidget(self.seq_preview)
        
            # Empaquetar en horizontal para que quede "la esquina derecha" ocupada por el preview
            row = QHBoxLayout()
            row.addWidget(seq_group, 2)
            row.addWidget(preview_group, 3)
        
            parent_layout.addLayout(row)

        _add_sequence_block_to_control_tab(self, bottom_layout)

        main_group_layout.addLayout(bottom_layout)

        # --- Add preconfiguration ---
        self.update_dom_config_button = QPushButton("Agregar configuración")
        main_group_layout.addWidget(self.update_dom_config_button)


        main_layout.addWidget(main_group)
        self.control_tab.setLayout(main_layout)

        #activate predet.
        # self.load_dom_config()

        # Conections
        self.saturate_dom_button.clicked.connect(self.saturate_dom)
        self.create_dom_button.clicked.connect(self.create_dom)
        self.update_dom_config_button.clicked.connect(self.update_dom_config)


    def iniciar_conexion(self):
        """
        Intenta conectar con los equipos vía PyVISA.
        Según los resultados, habilita los botones correspondientes.
        """
        rm = visa.ResourceManager()
        self.osci = None
        self.gen = None

        try:
            recursos = rm.list_resources()
            print(recursos)
            for resource in recursos:
                try:
                    instr = rm.open_resource(resource)
                    instr.timeout = 2000
                    IDN = instr.query("*IDN?")
                    print(f"{resource}:{IDN.strip()}")
                    instr.close()
            # print("Recursos detectados:", recursos)
                except Exception as e:
                    print(f"{resource}: No se pudo indentificar ({e})")

            self.osci = rm.open_resource("GPIB0::1::INSTR")
            self.fungen = rm.open_resource("GPIB0::10::INSTR")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo escanear recursos VISA:\n{e}")
            return

        # --- lógica de activación según qué se conectó ---
        if self.osci is not None and self.fungen is not None:
            estado = "✅ Se conectaron ambos equipos (osciloscopio y generador)."
            # Preconfiguramos el osciloscopio
            self.osci.write("ACQUIRE:MODE SAMPLE")
            self.osci.write("TRIG:A:MODE NORMAL")

            # Preconfiguramos el generador
            self.fungen.write("OUTP:LOAD INF")    # impedancia de salida: High Z
            # Configurar modo ráfaga (Burst Mode)
            self.fungen.write("BM:SOUR INT")      # fuente de ráfaga interna
            self.fungen.write("BM:NCYC 1")        # 1 ciclo por ráfaga
            self.fungen.write("BM:PHASe 0")       # fase inicial 0°
            self.fungen.write("BM:STAT ON")
            self._apply_burst_cycles_from_ui()# activar modo burst
            self.fungen.write("TRIG:SOUR BUS")    # trigger por software

            for btn in self.gen_buttons:
                btn.setEnabled(True)
            for btn in self.osc_buttons:
                btn.setEnabled(True)

        elif self.osci is not None:
            estado = "⚠️ Solo se conectó el osciloscopio."
            # Preconfiguramos el osciloscopio
            self.osci.write("ACQUIRE:MODE SAMPLE")
            self.osci.write("TRIG:A:MODE NORMAL")

            for btn in self.osc_buttons:
                btn.setEnabled(True)

        elif self.fungen is not None:
            estado = "⚠️ Solo se conectó el generador."
            # Preconfiguramos el generador
            self.fungen.write("OUTP:LOAD INF")    # impedancia de salida: High Z
            # Configurar modo ráfaga (Burst Mode)
            self.fungen.write("BM:SOUR INT")      # fuente de ráfaga interna
            self.fungen.write("BM:NCYC 1")        # 1 ciclo por ráfaga
            self.fungen.write("BM:PHASe 0")       # fase inicial 0°
            self.fungen.write("BM:STAT ON")       # activar modo burst
            self.fungen.write("TRIG:SOUR BUS")    # trigger por software

            for btn in self.gen_buttons:
                btn.setEnabled(True)

        else:
            estado = "❌ No se detectó ningún equipo."
            return QMessageBox.warning(self, "Conexión fallida", estado)

        QMessageBox.information(self, "Conexión completada", estado)

    def square_pulse(self,signo):
        n_puntos = 1000 #entre 8 y 16000 puntos soporta
        datos = np.zeros(n_puntos)
        datos[1:-2] = signo*1
        datos[0] = 0
        datos[-1] = 0
        return datos
    
    def sqr_osci_pulse(self,signo,A,f):
        w = 2*np.pi*f
        A = (A/2)/100
        n_puntos = 1000 #entre 8 y 16000 puntos soporta
        datos = np.zeros(n_puntos)
        t = np.linspace(0,1,len(datos[1: -2]))
        datos[1:-2] = signo*(1 + A*np.cos(w*t)-A)
        return datos
    
    def binarize_pulse(self,data):
        import struct
        # Escalado como dice el manual (1 V = 2047)
        datos_int = np.int16(np.clip(data * 2047,-2047,2047))
        # Convertir a binario (little endian para SWAP)
        binario = struct.pack('<' + 'h'*len(datos_int), *datos_int)
        
        # Crear encabezado SCPI para bloque binario
        bin_len = len(binario)
        bin_len_str = str(bin_len)
        header = f'DATA:DAC VOLATILE, #{len(bin_len_str)}{bin_len_str}'
        
        # Juntar todo (comando + binario) como un solo mensaje binario
        mensaje = header.encode('ascii') + binario
        
        return mensaje
    def _parse_int_or(self, text, default=1):
        try:
            return max(1, int(float(text)))
        except Exception:
            return default
    
    def _apply_burst_cycles_from_ui(self):
        if not hasattr(self, "fungen") or self.fungen is None:
            return
        ncyc = self._parse_int_or(self.nro_ciclo_edit.text(), 1)
        try:
            self.fungen.write(f"BM:NCYC {ncyc}")
            self.log_message(f"[GEN] BM:NCYC = {ncyc}")
        except Exception as e:
            self.log_message(f"[GEN][ERROR] No se pudo setear BM:NCYC: {e}")
    
    def _burst_enabled(self):
        try:
            resp = self.fungen.query("BM:STAT?").strip().upper()
            return ("1" in resp) or ("ON" in resp)
        except Exception:
            return False

    def ascii_pulse(self,datos):
        datos = np.clip(datos * 2047,-2047,2047).astype(int)
        data_str = ",".join(str(v) for v in datos)
        comando = "DATA:DAC VOLATILE," + data_str
        return comando

    def toggle_roi(self, text):
        self.roi_enabled = (text == "Sí")
        estado = "activado" if self.roi_enabled else "desactivado"
        self.log_message(f"Uso de ROI {estado}")
    
        # Mostrar inmediatamente la imagen con o sin ROI
        if self.last_full_image is not None:
            x = self.roi_x_input.value()
            y = self.roi_y_input.value()
            w = self.roi_width_input.value()
            h = self.roi_height_input.value()
    
            roi_valid = (
                self.roi_enabled and
                w >= 16 and h >= 16 and
                x + w <= self.last_full_image.shape[1] and
                y + h <= self.last_full_image.shape[0]
            )
    
            if roi_valid:
                image = self.last_full_image[y:y+h, x:x+w]
                self.log_message(f"[INFO] ROI aplicado en preview: x={x}, y={y}, w={w}, h={h}")
            else:
                image = self.last_full_image.copy()
                if self.roi_enabled:
                    self.log_message("[WARNING] ROI inválido en preview. Mostrando imagen completa.")
                else:
                    self.log_message("[INFO] ROI desactivado. Mostrando imagen completa.")
    
            self.display_captured_image_in_tab(image)


    def apply_roi_from_inputs(self):
        """
        Guarda los valores del ROI definidos manualmente,
        pero no los aplica a la cámara ni recorta la imagen hasta que se active el ROI.
        """
        x = self.roi_x_input.value()
        y = self.roi_y_input.value()
        width = self.roi_width_input.value()
        height = self.roi_height_input.value()
    
        if width % 8 != 0 or height % 8 != 0:
            self.log_message("[ERROR] El ancho y alto deben ser múltiplos de 8.")
            return
    
        self.log_message(f"ROI definido manualmente: x={x}, y={y}, w={width}, h={height} (no aplicado hasta que actives)")

    def _read_ui_value(self, candidates, cast=float, default=None):
        """Lee un valor de varios posibles widgets ya existentes para evitar redundancias.
        candidates: lista de nombres de atributos (QLineEdit/QSpinBox/QComboBox/etc.).
        Si ninguno existe, devuelve default.
        """
        for name in candidates:
            w = getattr(self, name, None)
            if w is None: continue
            try:
                if hasattr(w, 'value'):
                    return cast(w.value())
                if hasattr(w, 'text'):
                    txt = str(w.text()).replace(',', '.')
                    return cast(txt)
                if hasattr(w, 'currentText'):
                    txt = str(w.currentText()).replace(',', '.')
                    return cast(txt)
            except Exception:
                continue
        return default
    
    
    def _current_pulse_signature(self):
        """Arma la etiqueta de carpeta usando lo que ya hay en la UI.
        Ajustá los candidatos si usás otros nombres en tu código.
        """
        Hsat = self._read_ui_value(['sat_H_input','Hsat_input','satur_H_edit','H_sat','satH'], float, None)
        tsat = self._read_ui_value(['sat_t_input','tsat_input','satur_t_edit','t_sat','satT'], float, None)
        Hnuc = self._read_ui_value(['nuc_H_input','Hnuc_input','nucle_H_edit','H_nuc','nucH'], float, None)
        tnuc = self._read_ui_value(['nuc_t_input','tnuc_input','nucle_t_edit','t_nuc','nucT'], float, None)
        Hgro = self._read_ui_value(['cycle_H_input','Hgrow_input','grow_H_edit','H_ciclo','Hgrow'], float, None)
        tgro = self._read_ui_value(['cycle_t_input','Tgrow_input','grow_t_edit','t_ciclo','Tgrow'], float, None)
    
        parts = []
        if Hsat is not None and tsat is not None: parts.append(f"Sat{Hsat:g}Oe_{tsat:g}ms")
        if Hnuc is not None and tnuc is not None: parts.append(f"Nuc{Hnuc:g}Oe_{tnuc:g}ms")
        if Hgro is not None and tgro is not None: parts.append(f"Grow{Hgro:g}Oe_{tgro:g}ms")
        return "_".join(parts) if parts else "pulsos_actuales"
    
    
    def _roi_tuple_or_none(self):
        if getattr(self, 'roi_enabled', False):
            x = self.roi_x_input.value(); y = self.roi_y_input.value()
            w = self.roi_width_input.value(); h = self.roi_height_input.value()
            return (int(x), int(y), int(w), int(h))
        return None
    
    
    def _to_qpixmap(self, img16):
        arr = img16.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx <= mn:
            arr8 = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr8 = ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
        h, w = arr8.shape
        qimg = QImage(arr8.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())
    
    
    # -- E) Lanzar / detener secuencia --
    
    def _seq_run(self):
        if hasattr(self, 'seq_thread') and self.seq_thread is not None and self.seq_thread.isRunning():
            self.log_message("[SEQ] Ya hay una secuencia en ejecución.")
            return
        try:
            cycles = int(self.seq_cycles.value())
        except Exception:
            cycles = 1

        # Carpeta destino en work_dir (o cwd) con firma + timestamp
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        sign = self._current_pulse_signature()
        base_dir = self.work_dir if self.work_dir else os.getcwd()
        outdir = os.path.join(base_dir, f"SEQ_{sign}_{ts}")

        roi = self._roi_tuple_or_none()

        # Parámetros de resta de fondo y blur actuales (si usás otros nombres, ajustá aquí)
        bg_gain = getattr(self, 'background_gain', 1.0)
        bg_off  = getattr(self, 'background_offset', 0.0)
        blur_sigma = getattr(self, 'blur_strength', 0)
        capture_mode = getattr(self, 'capture_mode', 'Promedio')
        num_images = getattr(self, 'num_images_spinbox', None)
        if num_images is not None:
            num_images = max(1, int(num_images.value()))
        else:
            num_images = 1

        # Preparar Worker + hilo
        self.seq_thread = QThread()
        self.seq_worker = SequenceWorker(
            camera=self.camera,
            cycles=cycles,
            roi=roi,
            do_resta=True,
            background_gain=bg_gain,
            background_offset=bg_off,
            num_images_bg=num_images,
            num_images_frame=num_images,
            blur_sigma=blur_sigma,
            mode=capture_mode,
            outdir=outdir,
            parent_log=self._seq_threadsafe_log
        )
        self.seq_worker.moveToThread(self.seq_thread)

        # Conectar señales de control de pulsos a tus métodos existentes
        # Si tus métodos se llaman distinto, ajustá aquí. La idea: no duplicar lógica, reusar los ya conectados a tus botones.
        try:
            self.seq_worker.do_saturate.connect(self.saturate_dom)
        except Exception:
            pass
        try:
            self.seq_worker.do_nucleation.connect(self.create_dom)
        except Exception:
            pass
        # Crecimiento: si no tenés un método, podés implementar uno simple que lea la celda "características del ciclo" y dispare ese pulso.
        try:
            self.seq_worker.do_growth.connect(self.growth_pulse_from_cycle_cells)
        except Exception:
            # fallback: si no existe, repetimos create_dom como aproximación
            self.seq_worker.do_growth.connect(self.create_dom)
    
        # Señales de progreso/imagen
        self.seq_thread.started.connect(self.seq_worker.run)
        self.seq_worker.progress.connect(self._seq_on_progress)
        self.seq_worker.bg_ready.connect(self._seq_on_bg)
        self.seq_worker.image_ready.connect(self._seq_on_image)
        self.seq_worker.finished.connect(self._seq_on_finished)
        self.seq_worker.finished.connect(self.seq_thread.quit)
        self.seq_worker.finished.connect(self.seq_worker.deleteLater)
        self.seq_thread.finished.connect(self.seq_thread.deleteLater)

        # UI
        self.seq_run_btn.setEnabled(False)
        self.seq_stop_btn.setEnabled(True)
        self.seq_progress.setText(f"Guardando en: {outdir}")
    
        # Pausar preview si corresponde
        try:
            self.preview_worker.pause()
        except Exception:
            pass
    
        self.seq_thread.start()
        self.log_message("[SEQ] Iniciada")
    
    
    def _seq_stop(self):
        try:
            self.seq_worker.stop()
        except Exception:
            pass
        self.seq_stop_btn.setEnabled(False)
        self.log_message("[SEQ] Stop solicitado")
    
    
    def _seq_on_progress(self, pct, msg):
        self.seq_progress.setText(f"{pct}% — {msg}")
    
    
    def _seq_on_bg(self, bg_img):
        try:
            self.background_image = bg_img
        except Exception:
            pass
    
    
    def _seq_on_image(self, img_roi, idx):
        try:
            pix = self._to_qpixmap(img_roi)
            # Ajustar a tamaño del label manteniendo aspecto
            self.seq_preview.setPixmap(pix.scaled(
                self.seq_preview.width(), self.seq_preview.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        except Exception:
            pass
    
    
    def _seq_on_finished(self):
        self.seq_run_btn.setEnabled(True)
        self.seq_stop_btn.setEnabled(False)
        self.seq_progress.setText("Listo")
        try:
            self.preview_worker.resume()
        except Exception:
            pass
        try:
            if hasattr(self, 'seq_thread') and self.seq_thread is not None:
                self.seq_thread.wait()
        except Exception:
            pass
        self.seq_thread = None
        self.seq_worker = None
        self.log_message("[SEQ] Finalizada")

    def _seq_threadsafe_log(self, message):
        """Proxy para reenviar logs desde SequenceWorker al hilo principal."""
        self.seq_log_signal.emit(str(message))
    
    
    # -- F) (Opcional) Método para disparar el pulso de crecimiento leyendo la celda "características del ciclo" --
    # Si tu UI ya tiene un botón/método equivalente, borrá esto y conectá ese método arriba.
    
    def growth_pulse_from_cycle_cells(self):
        """Ejemplo genérico: arma un pulso usando las celdas de 'características del ciclo'.
        Ajustá los nombres de widgets si tus IDs reales difieren.
        """
        # Conversión H->V usando tus celdas ya existentes
        try:
            campo_corr = float(self.campo_corr_edit.text())
            resistencia = float(self.resistencia_edit.text())
        except Exception:
            self.log_message("[GROW] Faltan constante campo-corr y/o resistencia")
            return

        try:
            H = float(self.campo_ciclo_edit.text())
            t = float(self.tiempo_ciclo_edit.text())
        except Exception:
            self.log_message("[GROW] Ingresá campo y tiempo del ciclo válidos.")
            return
        if t <= 0:
            self.log_message("[GROW] El tiempo del ciclo debe ser positivo.")
            return

        if not getattr(self, 'fungen', None):
            self.log_message("[GROW] Generador no conectado.")
            return

        signo = -1 if self.radio_signo_pos_dom.isChecked() else +1

        # Misma fórmula que usás en Control
        corriente = H / campo_corr
        V = (corriente * resistencia / (10 * 0.95)) / 1000.0

        try:
            # SCPI mínimo: un ciclo cuadrado ~ t ms.
            frec = float(1000 / t)
            self.fungen.write('FREQ %f' % frec)
            time.sleep(0.02)
            self.fungen.write('VOLT:OFFS 0')
            time.sleep(0.02)
            self.fungen.write('VOLT %f' % V)
            time.sleep(0.02)

            pulse_type = self.combo_pulso_ciclo.currentText().strip().lower()
            if pulse_type == "pulso cuadrado":
                pulso = self.square_pulse(signo)
            elif pulse_type == "pulso oscilatorio":
                try:
                    A = float(self.amp_edit.text())
                    f = float(self.frec_edit.text())
                except Exception:
                    self.log_message("[GROW] Parámetros de pulso oscilatorio inválidos; usando cuadrado.")
                    pulso = self.square_pulse(signo)
                else:
                    pulso = self.sqr_osci_pulse(signo, A, f)
            elif pulse_type == "pulso mixto":
                self.log_message("[GROW] Pulso mixto no implementado; usando cuadrado.")
                pulso = self.square_pulse(signo)
            else:
                pulso = self.square_pulse(signo)


            binario = self.binarize_pulse(pulso)
            # comando = self.ascii_pulse(pulso)
            self.fungen.write('FORM:BORD SWAP')
            time.sleep(0.1)
    
            self.fungen.write_raw(binario)
            # self.fungen.write(comando)
    
            # Seleccionar y activar la forma de onda descargada
            self.fungen.write('FUNC:USER VOLATILE')
            # print(self.fungen.query("FUNC:USER?"))
            self.fungen.write('FUNC:SHAP USER')
            # print(self.fungen.query("FUNC:SHAP?"))
            self.fungen.write('VOLT %f' % V)
            # print(self.fungen.query("VOLT?"))
            self.fungen.write('*TRG')


        except Exception as e:
            self.log_message(f"[GROW][ERROR] {e}")
            
    def apply_real_camera_values_to_sliders(self):
        """
        Lee los valores reales de cámara (o defaults en simulación)
        y sincroniza sliders + inputs (escala x100).
        """
        for prop, (_, _, default) in self.properties.items():
            try:
                value = default if self.simulation else self.camera.GetProperty(prop)[0]
            except Exception:
                value = default
            self.sliders[prop].blockSignals(True)
            self.sliders[prop].setValue(int(value * 100))
            self.sliders[prop].blockSignals(False)
            self.inputs[prop].setText(f"{value:.2f}")

        
    def update_property(self, prop, value):
        """
        Updates a given camera property both in the camera object
        and synchronizes the value across the GUI slider and input field.
    
        Parameters:
            prop (str): property name (e.g., 'brightness').
            value (float): new value to assign.
        """
        if self.camera:
            self.camera.set_properties(**{prop: value})
    
        # Actualiza sliders y textbox
        self.sliders[prop].blockSignals(True)
        self.sliders[prop].setValue(int(value * 100))
        self.sliders[prop].blockSignals(False)
        self.inputs[prop].setText(f"{value:.2f}")
        self.log_message(f"Se actualizó '{prop}' a {value:.2f}")
    
        #Captura una imagen nueva inmediatamente para reflejar el cambio
        if self.preview_mode:
            try:
                self.preview_worker.resume()
            except Exception as e:
                self.log_message(f"[ERROR] No se pudo actualizar el preview tras cambiar {prop}: {e}")


    def set_roi(self, width, height, x_offset=0, y_offset=0):
        """
        Reduces the ROI (Region of Interest) by setting a smaller width and height.
        This speeds up the readout time of the camera.
    
        Parameters:
            width (int): desired width of ROI (must be multiple of 8).
            height (int): desired height of ROI (must be multiple of 8).
            x_offset (int): horizontal offset (default 0).
            y_offset (int): vertical offset (default 0).
        """
        try:
            frameformat, fps = self.camera.GetFormat()
            frameformat.width = width
            frameformat.height = height
            frameformat.xOffset = x_offset
            frameformat.yOffset = y_offset
            self.camera.SetFormat(frameformat, fps)
            self.log_message(f"ROI actualizado a {width}x{height} desde ({x_offset},{y_offset})")
        except Exception as e:
            self.log_message(f"[ERROR] No se pudo aplicar ROI: {e}")        
        
    def change_fps(self, fps_text):
        try:
            fps = float(fps_text)
            frameformat, _ = self.camera.GetFormat()
            self.camera.SetFormat(frameformat, fps)
            self.log_message(f"FPS cambiado a {fps:.2f}")
        except Exception as e:
            self.log_message(f"[ERROR] No se pudo cambiar el FPS: {e}")

    def set_roi_from_mouse(self, x, y, w, h):
        """
        Callback triggered when user selects a region with the mouse.
        Updates the spinboxes and applies the ROI to the camera.
        """
        # Get current camera format
        frameformat, fps = self.camera.GetFormat()
        max_width = frameformat.width
        max_height = frameformat.height
        
        # Calculate scaling factor if preview is resized
        scale_x = max_width / self.preview_label_capture.width()
        scale_y = max_height / self.preview_label_capture.height()
        
        # Scale coordinates to camera resolution
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        
        # Round to multiples of 8
        w = (w // 8) * 8
        h = (h // 8) * 8
        x = (x // 8) * 8
        y = (y // 8) * 8
        
        # Validate bounds
        x = max(0, min(x, max_width - 8))
        y = max(0, min(y, max_height - 8))
        w = min(w, max_width - x)
        h = min(h, max_height - y)
        
        self.roi_x_input.setValue(x)
        self.roi_y_input.setValue(y)
        self.roi_width_input.setValue(max(w, 8))
        self.roi_height_input.setValue(max(h, 8))
        
        self.log_message(f"ROI selected: x={x}, y={y}, w={w}, h={h}")

    def set_property_from_input(self, prop, field):
        """
        Parses a float from the input field and updates the associated property.
        Reverts to slider value if parsing fails.
        
        Parameters:
            prop (str): property name.
            field (QLineEdit): corresponding input widget.
        """
        try:
            value = float(field.text())
            self.sliders[prop].setValue(int(value * 100))
            self.camera.set_properties(**{prop: value})
        except ValueError:
            field.setText(f"{self.sliders[prop].value() / 10:.1f}")
    
    def update_blur(self, value):
        """
        Updates blur strength and syncs slider/input values.
        
        Parameters:
            value (int): blur level (0–10).
        """
        self.blur_strength = value
        self.blur_label.setText(f"Blur: {value}")
        self.blur_input.setText(str(value))
        self.log_message(f"Se seteó el blur a {value}")
    
    def set_blur_from_input(self):
        """
        Parses blur value from input, validates range, and updates slider.
        Reverts to slider if input is invalid.
        """
        try:
            value = int(self.blur_input.text())
            if 0 <= value <= 10:
                self.blur_slider.setValue(value)
                self.log_message(f"Se ingresó blur manualmente a {value}")
            else:
                self.blur_input.setText(str(self.blur_slider.value()))
        except ValueError:
            self.blur_input.setText(str(self.blur_slider.value()))

    def update_gain(self, value):
        """
        Updates background subtraction gain (scaling factor).

        Parameters:
            value (float): gain multiplier.
        """
        self.background_gain = value
        self.gain_input.setText(f"{value:.2f}")
        self.log_message(f"Ganancia del fondo ajustada a {value:.2f}")

    def set_gain_from_input(self):
        """
        Sets gain from the input field.
        Reverts to current gain if input is invalid.
        """
        try:
            value = float(self.gain_input.text())
            self.background_gain = value
            self.gain_slider.setValue(int(value * 100))
        except ValueError:
            self.gain_input.setText(f"{self.background_gain:.2f}")

    def update_offset(self, value):
        """
        Updates background subtraction offset.

        Parameters:
            value (int): pixel-wise bias to subtract before scaling.
        """
        self.background_offset = value * 32768  # Mapear de [-1,1] a [-32768,32768]
        self.offset_input.setText(f"{value:.2f}")
        self.log_message(f"Offset del fondo ajustado a {self.background_offset:.0f} (escalado: {value:.2f})")

    def set_offset_from_input(self):
        """
        Parses and sets offset value from user input field.
        If invalid, reverts to current value.
        """
        try:
            value = float(self.offset_input.text())
            if -1.0 <= value <= 1.0:
                self.background_offset = value * 32768
                self.offset_slider.setValue(int(value * 100))
            else:
                self.offset_input.setText(f"{self.background_offset / 32768:.2f}")
        except ValueError:
            self.offset_input.setText(f"{self.background_offset / 32768:.2f}")
            
    def change_capture_mode(self, mode):
        """
        Updates internal state when user selects capture mode.

        Parameters:
            mode (str): either 'Promedio' or 'Mediana'.
        """
        self.capture_mode = mode
        self.log_message(f"Modo de captura cambiado a {mode}")
    
    def select_work_dir(self):
        """
        Opens a dialog to choose the directory for saving images.
        Sets self.work_dir and updates input field.
        """
        directory = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio de Trabajo")
        if directory:
            self.work_dir = directory
            self.dir_line_edit.setText(directory)
            self.log_message(f"Directorio de trabajo establecido: {directory}")
    
    def toggle_auto_save(self, text):
        """
        Enables or disables automatic saving of images.

        Parameters:
            text (str): "Sí" to enable, "No" to disable.
        """
        self.auto_save = (text == "Sí")
        estado = "activado" if self.auto_save else "desactivado"
        self.log_message(f"Guardado automático {estado}")
    
    def save_auto_images(self, full_image, roi_image, roi_coords):
        x, y, w, h = roi_coords
        """Save all automatic image versions"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
        # 1. Save raw full image (no ROI, no processing)
        raw_folder = os.path.join(self.work_dir, "raw")
        os.makedirs(raw_folder, exist_ok=True)
        raw_path = os.path.join(raw_folder, f"cruda_{timestamp}.tif")
        imsave(raw_path, full_image)
        self.log_message(f"Imagen cruda (completa) guardada en: {raw_path}")
    
        # 2. Save normal image (ROI + optional blur, NO background subtraction)
        normal_image = roi_image
        if self.blur_strength > 0:
            normal_image = blur_uint16(normal_image, sigma=self.blur_strength)
        
        normal_path = os.path.join(self.work_dir, f"normal_{timestamp}.tif")
        imsave(normal_path, normal_image)
        self.log_message(f"Imagen normal (ROI + blur) guardada en: {normal_path}")
    
        # 3. Save resta image (ROI + blur + background subtraction)
        if self.background_enabled and self.background_image is not None:
            resta_image = self.apply_background_subtraction(
                normal_image, x, y, w, h
            )
            resta_path = os.path.join(self.work_dir, f"resta_{timestamp}.tif")
            imsave(resta_path, resta_image)
            self.log_message(f"Imagen resta (ROI + blur + fondo) guardada en: {resta_path}")
        else:
            self.log_message("Fondo no activado o no disponible, imagen 'resta' no guardada.")
            
    def capture_background(self):
        """
        Initiates the background image acquisition process.
        Pauses the preview worker and starts a Worker instance
        in a separate thread to compute the background image.
        """
        self.preview_worker.pause()
        self.preview_mode = False
        num_images = self.num_images_spinbox.value()
        mode = self.capture_mode_selector.currentText()
        blur_strength = self.blur_strength
        self.background_worker = Worker(self.camera, num_images, mode, blur_strength)
        self.background_worker.image_captured.connect(self.set_background_image)
        threading.Thread(target=self.background_worker.run, daemon=True).start()
        self.log_message("Iniciando captura de fondo...")
    
    def set_background_image(self, image):
        """
        Assigns the captured background image to internal state,
        displays it, and optionally saves it.

        Parameters:
            image (np.ndarray): background image to store and show.
        """
        self.background_image = image
        self.display_image(image)
        self.log_message("Fondo capturado correctamente.")
        if self.auto_save and self.work_dir:
            fondo_path = os.path.join(self.work_dir, f"fondo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tif")
            imsave(fondo_path, image)
            self.log_message(f"Imagen de fondo guardada en: {fondo_path}")
    
    def toggle_background(self, text):
        """
        Enables or disables background subtraction mode.

        Parameters:
            text (str): "Sí" to enable, "No" to disable.
        """
        self.background_enabled = (text == "Sí")
        estado = "activada" if self.background_enabled else "desactivada"
        self.log_message(f"Restar fondo {estado}")

    def apply_background_subtraction(self, image_roi, x=0, y=0, w=None, h=None):
        """
        Applies background subtraction using a cropped ROI from the stored background.
        """
        if (
            self.background_image is None or
            w is None or h is None or
            self.background_image.shape[0] < y + h or
            self.background_image.shape[1] < x + w
        ):
            return image_roi
    
        bg_roi = self.background_image[y:y+h, x:x+w]
        a = self.background_gain
        b = self.background_offset
    
        image_float = image_roi.astype(np.float32)
        bg_float = bg_roi.astype(np.float32)
    
        diff = a * (image_float - bg_float + b)
        
        # Reescalar a [0, 65535]
        min_val = diff.min()
        max_val = diff.max()
        diff_16bits = ((diff - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
        
        return diff_16bits
    
    def capture_image(self):
        """
        Starts the acquisition of the main image using Worker.
        Uses user-selected mode and blur. If in simulation mode,
        logs appropriate note.
        """
        self.preview_worker.pause()
        num_images = self.num_images_spinbox.value()
        mode = self.capture_mode_selector.currentText()
        if self.simulation:
            self.log_message("Captura de imagen simulada.")
    
        self.worker = Worker(self.camera, num_images, mode, self.blur_strength)
        self.worker.image_captured.connect(self.display_captured_image)
        threading.Thread(target=self.worker.run, daemon=True).start()
        self.log_message("Iniciando captura de imagen...")
        
    def display_captured_image(self, image):
        """
        Handles the display of a newly captured image.
        Applies optional background subtraction with gain/offset
        and sets GUI buttons accordingly.
    
        Parameters:
            image (np.ndarray): final processed image.
        """
        # Store the original full image without any processing
        self.last_full_image = image.copy()
        
        # Apply ROI if enabled
        x = self.roi_x_input.value()
        y = self.roi_y_input.value()
        w = self.roi_width_input.value()
        h = self.roi_height_input.value()
    
        roi_valid = (
            self.roi_enabled and
            w >= 16 and h >= 16 and
            x + w <= image.shape[1] and
            y + h <= image.shape[0]
        )
    
        if roi_valid:
            image_roi = image[y:y+h, x:x+w]
            self.log_message(f"[INFO] ROI aplicado: x={x}, y={y}, w={w}, h={h}")
        else:
            image_roi = image.copy()
            if self.roi_enabled:
                self.log_message("[WARNING] ROI inválido o trivial. Mostrando imagen completa.")
            else:
                self.log_message("[INFO] ROI desactivado. Mostrando imagen completa.")
                
        if not roi_valid:
            x, y = 0, 0
            h, w = image.shape

    
        # Store the ROI-applied image
        self.captured_image = image_roi
    
        # Apply background subtraction if needed (only for display)
        if self.background_enabled and self.background_image is not None:
            display_image = self.apply_background_subtraction(image_roi, x, y, w, h)
        else:
            display_image = image_roi
    
        # Show in interface
        self.display_captured_image_in_tab(display_image)
        self.save_button.setEnabled(True)
        self.toggle_preview_button.setEnabled(True)
        self.log_message("Imagen capturada y mostrada en pestaña 'Captura'.")
    
        # Auto-save different versions
        if self.auto_save:
            self.save_auto_images(
                full_image=image.copy(),
                roi_image=image_roi.copy(),
                roi_coords=(x, y, w, h)
            )
        
        self.last_processed_image = image.copy()

            
    def display_captured_image_in_tab(self, image, scale_factor=1):
        if len(image.shape) == 3:
            image = (rgb2gray(image) * 65535).astype(np.uint16)
    

        height, width = image.shape
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
    
        resized_image = resize(image, (new_height, new_width), preserve_range=True).astype(np.uint16)
        image_8bit = to_8bit_for_preview(resized_image)
        image_8bit = np.ascontiguousarray(image_8bit)
    
        bytes_per_line = image_8bit.shape[1]
        qimage = QImage(image_8bit.data, image_8bit.shape[1], image_8bit.shape[0], bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.preview_label_capture.width(), 
            self.preview_label_capture.height(), 
            Qt.KeepAspectRatio
        )
        self.preview_label_capture.setPixmap(pixmap)
    

    def save_image(self):
        """
        Opens a dialog to let user save the current captured image manually.
        If no image is available, logs error.
        """
        if self.captured_image is None:
            self.log_message("[ERROR] No hay imagen capturada para guardar.")
            return
        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix("tif")
        file_path, _ = file_dialog.getSaveFileName(self, "Guardar Imagen", "", "TIFF (*.tif)")
        if file_path:
            imsave(file_path, self.captured_image)
            self.log_message(f"Imagen guardada manualmente en: {file_path}")
    
    def save_preview_parameters(self):
        """
        Saves current preview tab parameters (camera properties + FPS) to a JSON file.
        """
        params = {
            prop: self.sliders[prop].value() / 10 for prop in self.properties
        }
        # Add FPS
        try:
            current_fps = float(self.fps_selector.currentText())
            params["fps"] = current_fps
        except Exception as e:
            self.log_message(f"[WARNING] Could not save FPS: {e}")
    
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Preview Parameters", "", "JSON (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(params, f, indent=4)
                self.log_message(f"Preview parameters saved to {file_path}")
            except Exception as e:
                self.log_message(f"[ERROR] Could not save preview parameters: {e}")

    def save_capture_parameters(self):
        """
        Saves current capture-related settings to JSON (blur, mode, count).
        """
        params = {
            "blur_strength": self.blur_strength,
            "num_images": self.num_images_spinbox.value(),
            "capture_mode": self.capture_mode,
            "background_gain": self.background_gain,
            "background_offset": self.background_offset,
            "roi_x": self.roi_x_input.value(),
            "roi_y": self.roi_y_input.value(),
            "roi_width": self.roi_width_input.value(),
            "roi_height": self.roi_height_input.value(),
        }
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Parámetros de Captura", "", "JSON (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(params, f, indent=4)
                self.log_message(f"Parámetros de Captura guardados en {file_path}")
            except Exception as e:
                self.log_message("[ERROR] No se pudieron guardar parámetros de Captura: {e}")
                
    def save_parameters(self):
        """
        Saves parameters depending on the active tab (preview or capture).
        """
        if self.tabs.currentIndex() == 0:
            self.save_preview_parameters()
        else:
            self.save_capture_parameters()                


    def load_parameters(self):
        """
        Loads both preview and capture parameters from a JSON file,
        and applies the values to appropriate widgets.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Cargar Parámetros", "", "JSON (*.json)")
        if file_path:
            with open(file_path, 'r') as f:
                params = json.load(f)
    
            # Preview properties
            for prop in self.properties:
                value = params.get(prop, None)
                if value is not None:
                    self.sliders[prop].setValue(int(value))
                    self.update_property(prop, value)
    
            # FPS
            if 'fps' in params:
                fps_str = f"{params['fps']:.2f}"
                index = self.fps_selector.findText(fps_str)
                if index != -1:
                    self.fps_selector.setCurrentIndex(index)
                else:
                    self.log_message(f"[WARNING] Saved FPS ({fps_str}) not in available list.")
    
            # Capture settings
            if 'blur_strength' in params:
                self.blur_slider.setValue(params['blur_strength'])
                self.update_blur(params['blur_strength'])
    
            if 'num_images' in params:
                self.num_images_spinbox.setValue(params['num_images'])
    
            if 'background_gain' in params:
                self.update_gain(params['background_gain'])
    
            if 'background_offset' in params:
                self.update_offset(params['background_offset'])
    
            if 'capture_mode' in params:
                index = self.capture_mode_selector.findText(params['capture_mode'])
                if index != -1:
                    self.capture_mode_selector.setCurrentIndex(index)
                    
            if 'roi_x' in params:
                self.roi_x_input.setValue(params['roi_x'])
            if 'roi_y' in params:
                self.roi_y_input.setValue(params['roi_y'])
            if 'roi_width' in params:
                self.roi_width_input.setValue(params['roi_width'])
            if 'roi_height' in params:
                self.roi_height_input.setValue(params['roi_height'])
            self.apply_roi_from_inputs()


            self.log_message(f"Parámetros cargados desde archivo: {file_path}")

    
    def start_preview(self):
        """
        Resumes live preview if previously paused.
        """
        self.preview_mode = True
        self.toggle_preview_button.setEnabled(False)
        self.preview_worker.resume()
        self.log_message("Preview reanudado.")

    def display_preview_image(self, image, scale_factor=1):
        """
        Displays a live preview image in the preview panel.
        Converts to 8-bit, scales if needed.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = (rgb2gray(image) * 65535).astype(np.uint16)
    
        # Resize if needed
        if scale_factor != 1:
            height, width = image.shape
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = resize(image, (new_height, new_width), preserve_range=True).astype(np.uint16)
    
        # Convert to 8-bit for display
        image_8bit = to_8bit_for_preview(image)
        qimage = QImage(image_8bit.data, image_8bit.shape[1], image_8bit.shape[0],
                        image_8bit.shape[1], QImage.Format_Grayscale8)
    
        # Update preview display
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.preview_label_preview.width(), 
            self.preview_label_preview.height(), 
            Qt.KeepAspectRatio
        )
        self.preview_label_preview.setPixmap(pixmap)


    def display_image(self, image, scale_factor=1):
        """
        Updates both preview and capture image panels with given image.

        Parameters:
            image (np.ndarray): grayscale image.
            scale_factor (float): resize factor.
        """
        if len(image.shape) == 3:
            image = (rgb2gray(image) * 65535).astype(np.uint16)
    
        height, width = image.shape
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
    
        resized_image = resize(image, (new_height, new_width), preserve_range=True).astype(np.uint16)
        image_8bit = to_8bit_for_preview(resized_image)
    
        bytes_per_line = image_8bit.shape[1]
        qimage = QImage(image_8bit.data, image_8bit.shape[1], image_8bit.shape[0], bytes_per_line, QImage.Format.Format_Grayscale8)
    
        self.preview_label_preview.setPixmap(QPixmap.fromImage(qimage))
        self.preview_label_capture.setPixmap(QPixmap.fromImage(qimage))
        
    def get_last_image(self):
        """
        Devuelve la última imagen capturada en 16 bits, o None si no hay.
        """
        return getattr(self, "last_processed_image", None)
    
    def create_dom(self):
        """
        Lee los valores de los campos y tiempos de creación de dominio desde la interfaz,
        calcula la corriente correspondiente al campo deseado,
        y muestra la información o la envía al generador de pulsos.
        """
        # --- 0. Verificar que los campos no estén vacíos ---
        if (not self.campo_dominio_edit.text().strip() or
            not self.tiempo_dominio_edit.text().strip() or
            not self.campo_corr_edit.text().strip() or
            not self.resistencia_edit.text().strip()):
            QMessageBox.warning(self, "Campos incompletos", "Completá todos los campos numéricos antes de continuar: \n"
                                "Campo \n"
                                "Tiempo\n"
                                "Relación campo-corriente\n"
                                "Resistencia")
            return

        # --- 1. Verificar que algún signo esté seleccionado ---
        if not (self.radio_signo_pos_dom.isChecked() or self.radio_signo_neg_dom.isChecked()):
            QMessageBox.warning(self, "Signo no seleccionado", "Seleccioná el signo del pulso antes de continuar.")
            return

        try:
            # --- 2. Leer parámetros de saturación ---
            campo_saturacion = float(self.campo_dominio_edit.text())  # [Oe]
            tiempo_saturacion = float(self.tiempo_dominio_edit.text())  # [ms]

            # --- 3. Leer signo seleccionado ---
            signo = -1 if self.radio_signo_pos_dom.isChecked() else +1  #Cambio el signo para que nuclee en la dirección pedida (OPAMP inversor)

            # --- 4. Leer relación campo-corriente y resistencia ---
            campo_corr = float(self.campo_corr_edit.text())  # [Oe/A]
            resistencia = float(self.resistencia_edit.text())  # [Ohm]

            # --- 5. Calcular corriente y tensión necesarias ---
            corriente = campo_saturacion / campo_corr       # [A]
            tension = float((corriente * resistencia/(10*0.95))/1000) 
            print(tension)                      # [V]           
            #dividimos por 10 para tener la tensión enviada por el generador (estamos viendo la del OPAMP) y se tiene en cuenta una caida
            # del 5% respecto de lo enviado vía digital a lo medido realmente.
        except ValueError:
            QMessageBox.warning(self, "Error de entrada", "Verificá que todos los valores sean numéricos.")

        if self.fungen.query("BM:STAT?") == 0:
            self.iniciar_conexion
            QMessageBox.warning(self, "Error", "El equipo no esta en modo Ráfaga.")
            return
        frec = float(1000/tiempo_saturacion)
        self.fungen.write('FREQ %f' % frec)
        time.sleep(0.1)
        # print(self.fungen.query("FREQ?"))
        self.fungen.write('VOLT:OFFS 0')
        time.sleep(0.1)
        # print(self.fungen.query("VOLT:OFFS?"))
        self.fungen.write('VOLT %f' % tension)
        time.sleep(0.1)
        
        pulso = self.square_pulse(signo)        
        binario = self.binarize_pulse(pulso)
        # comando = self.ascii_pulse(pulso)
        self.fungen.write('FORM:BORD SWAP')
        time.sleep(0.1)

        self.fungen.write_raw(binario)
        # self.fungen.write(comando)

        # Seleccionar y activar la forma de onda descargada
        self.fungen.write('FUNC:USER VOLATILE')
        # print(self.fungen.query("FUNC:USER?"))
        self.fungen.write('FUNC:SHAP USER')
        # print(self.fungen.query("FUNC:SHAP?"))
        self.fungen.write('VOLT %f' % tension)
        # print(self.fungen.query("VOLT?"))
        self.fungen.write('*TRG')

    def saturate_dom(self):
        """
        Lee los valores de los campos y tiempos de saturación desde la interfaz,
        calcula la corriente correspondiente al campo deseado,
        y muestra la información o la envía al generador de pulsos.
        """
        # --- 0. Verificar que los campos no estén vacíos ---
        if (not self.campo_saturacion_edit.text().strip() or
            not self.tiempo_saturacion_edit.text().strip() or
            not self.campo_corr_edit.text().strip() or
            not self.resistencia_edit.text().strip()):
            QMessageBox.warning(self, "Campos incompletos", "Completá todos los campos numéricos antes de continuar: \n"
                                "Campo \n"
                                "Tiempo\n"
                                "Relación campo-corriente\n"
                                "Resistencia")
            return

        # --- 1. Verificar que algún signo esté seleccionado ---
        if not (self.radio_signo_pos_dom.isChecked() or self.radio_signo_neg_dom.isChecked()):
            QMessageBox.warning(self, "Signo no seleccionado", "Seleccioná el signo del pulso antes de continuar.")
            return

        try:
            # --- 2. Leer parámetros de saturación ---
            campo_saturacion = float(self.campo_saturacion_edit.text())  # [Oe]
            tiempo_saturacion = float(self.tiempo_saturacion_edit.text())  # [ms]

            # --- 3. Leer signo seleccionado ---
            signo = +1 if self.radio_signo_pos_dom.isChecked() else -1

            # --- 4. Leer relación campo-corriente y resistencia ---
            campo_corr = float(self.campo_corr_edit.text())  # [Oe/A]
            resistencia = float(self.resistencia_edit.text())  # [Ohm]

            # --- 5. Calcular corriente y tensión necesarias ---
            corriente = campo_saturacion / campo_corr       # [A]
            tension = float((corriente * resistencia/(10*0.95))/1000) 
            print(tension)                      # [V]           
            #dividimos por 10 para tener la tensión enviada por el generador (estamos viendo la del OPAMP) y se tiene en cuenta una caida
            # del 5% respecto de lo enviado vía digital a lo medido realmente.
        except ValueError:
            QMessageBox.warning(self, "Error de entrada", "Verificá que todos los valores sean numéricos.")

        if self.fungen.query("BM:STAT?") == 0:
            self.iniciar_conexion
            QMessageBox.warning(self, "Error", "El equipo no esta en modo Ráfaga.")
            return
        
        frec = float(1000/tiempo_saturacion)
        self.fungen.write('FREQ %f' % frec)
        time.sleep(0.1)
        # print(self.fungen.query("FREQ?"))
        self.fungen.write('VOLT:OFFS 0')
        time.sleep(0.1)
        # print(self.fungen.query("VOLT:OFFS?"))
        self.fungen.write('VOLT %f' % tension)
        time.sleep(0.1)
        print(self.fungen.query('VOLT?'))
        
        pulso = self.square_pulse(signo)        
        binario = self.binarize_pulse(pulso)
        # comando = self.ascii_pulse(pulso)
        self.fungen.write('FORM:BORD SWAP')
        time.sleep(0.1)

        self.fungen.write_raw(binario)
        # self.fungen.write(comando)
        print(self.fungen.query('VOLT?'))
                
        # Seleccionar y activar la forma de onda descargada
        self.fungen.write('FUNC:USER VOLATILE')
        time.sleep(0.1)
        print(self.fungen.query('VOLT?'))
        # print(self.fungen.query("FUNC:USER?"))
        self.fungen.write('FUNC:SHAP USER')
        time.sleep(0.1)
        print(self.fungen.query('VOLT?'))
        self.fungen.write('VOLT %f' % tension)
        time.sleep(0.1)
        print(self.fungen.query('VOLT?'))
        # print(self.fungen.query("FUNC:SHAP?"))
        self.fungen.write('*TRG')


    def load_dom_config(self, nombre = "PREDETERMINADA"):
        # Verificar existencia del JSON
        file_name = "../../params/params_preconfiguration.json"
        if not os.path.exists(file_name):
            QMessageBox.warning(self, "Error", f"El archivo '{file_name}' no existe.")
            return None
        
        with open(file_name, "r", encoding="utf-8") as f:
            datos = json.load(f)

        if nombre not in datos:
            QMessageBox.warning(self, "Error", f"No se encontró la configuración '{nombre}' en el archivo JSON.")
            return

        config = datos[nombre]

        # -----------------------------
        # 4. Cargar los valores en los QLineEdit
        # -----------------------------
        self.tiempo_saturacion_edit.setText(config.get("tiempo_saturacion", ""))
        self.campo_saturacion_edit.setText(config.get("campo_saturacion", ""))
        self.tiempo_dominio_edit.setText(config.get("tiempo_dominio", ""))
        self.campo_dominio_edit.setText(config.get("campo_dominio", ""))
        self.resistencia_edit.setText(config.get("resistencia", ""))
        self.campo_corr_edit.setText(config.get("campo-corr", ""))

        # (Opcional) mensaje de confirmación
        # QMessageBox.information(self, "Configuración cargada", f"Se cargaron los valores de '{nombre}'.")

    def update_dom_config(self):
        '''updates a .json file with data that the user wants'''
        # Verificar existencia del JSON
        file_name = "../../params/params_preconfiguration.json"
        if not os.path.exists(file_name):
            QMessageBox.warning(self, "Error", f"El archivo '{file_name}' no existe.")
            return None
        
        with open(file_name, "r", encoding="utf-8") as f:
            datos = json.load(f)
        
        # Consultar nombre de la nueva configuración
        nombre, ok = QInputDialog.getText(self, "Nueva configuración", "Ingrese nombre de configuración:")
        if not ok or not nombre.strip():
            return  # usuario canceló
        
         # Verificar si la clave ya existe
        if nombre in datos:
            QMessageBox.warning(self, "Error", f"La clave '{nombre}' ya existe en el JSON.")
            return

        # Crear diccionario con los datos de los QLineEdit
        nueva_info = {
                "tiempo_saturacion": self.tiempo_saturacion_edit.text(),
                "campo_saturacion": self.campo_saturacion_edit.text(),
                "tiempo_dominio": self.tiempo_dominio_edit.text(),
                "campo_dominio": self.campo_dominio_edit.text(),
                "resistencia": self.resistencia_edit.text(),
                "campo-corr": self.campo_corr_edit.text(),
            }
        
        # Agregar nueva configuración al diccionario
        datos[nombre] = nueva_info

        # Guardar de nuevo el JSON
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)
        
        # Actualizar combobox
        self.combo.clear()
        for nombre_config, valores in datos.items():
            self.combo.addItem(nombre_config, valores)


    def log_message(self, message):
        """
        Logs a message with timestamp to both GUI consoles and log.txt.

        Parameters:
            message (str): message content.
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
    
        self.console_preview.appendPlainText(full_message)
        self.console_capture.appendPlainText(full_message)
    
        print(full_message)  # Para consola de Spyder o terminal
    
        # También guardar en log.txt
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.write(full_message + "\n")
            self.log_file.flush()

    def closeEvent(self, event):
        """
        Handles application exit: stops threads and closes logs.
        """
        self.preview_worker.stop()
        self.preview_thread.quit()
        self.preview_thread.wait()
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()
        event.accept()



# Main execution block
if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())


    