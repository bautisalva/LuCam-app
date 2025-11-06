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
                             QSpinBox, QDoubleSpinBox, QMessageBox, QSizePolicy)
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


def square_pulse(signo):
    n_puntos = 1000  # entre 8 y 16000 puntos soporta
    datos = np.zeros(n_puntos)
    datos[1:-2] = signo * 1
    datos[0] = 0
    datos[-1] = 0
    return datos


def sqr_osci_pulse(signo, amplitud_porcentaje, oscilaciones):
    w = 2 * np.pi * oscilaciones
    amplitud = (amplitud_porcentaje / 2) / 100
    n_puntos = 1000  # entre 8 y 16000 puntos soporta
    datos = np.zeros(n_puntos)
    t = np.linspace(0, 1, len(datos[1:-2]))
    datos[1:-2] = signo * (1 + amplitud * np.cos(w * t) - amplitud)
    return datos


def triangular_pulse(signo, amplitud_porcentaje, triangulos, geometria):
    """Genera un pulso triangular normalizado para el generador."""

    n_puntos = 1000
    datos = np.zeros(n_puntos)
    interior = datos[1:-2]
    M = len(interior)
    if M <= 0:
        return datos

    triangulos = max(1, int(triangulos))
    amplitud = max(0.0, float(amplitud_porcentaje)) / 100.0
    eps = 1e-6
    geometria = float(np.clip(geometria, eps, 1 - eps))

    t = np.linspace(0.0, float(triangulos), M, endpoint=False)
    phi = t % 1.0
    subida = -1.0 + 2.0 * (phi / geometria)
    bajada = 1.0 - 2.0 * ((phi - geometria) / (1.0 - geometria))
    onda = np.where(phi < geometria, subida, bajada)
    interior[:] = signo * (1.0 + onda * amplitud)
    return datos


def binarize_pulse(data):
    import struct

    datos_int = np.int16(np.clip(data * 2047, -2047, 2047))
    binario = struct.pack('<' + 'h' * len(datos_int), *datos_int)

    bin_len = len(binario)
    bin_len_str = str(bin_len)
    header = f'DATA:DAC VOLATILE, #{len(bin_len_str)}{bin_len_str}'

    mensaje = header.encode('ascii') + binario
    return mensaje


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
    image_ready = pyqtSignal(np.ndarray, int)
    finished = pyqtSignal()

    def __init__(self, *, camera, fungen, satur_pulse, nucleation_pulse,
                 growth_pulse, cycles, roi, num_images_bg, num_images_frame,
                 outdir, campo_corr, resistencia, blur_sigma=0,
                 do_resta=True, mode="Promedio", parent_log=None,
                 background_gain=1.0, background_offset=0.0,
                 sequence_mode="completa", repetitions=1):
        super().__init__()
        self.camera = camera
        self.fungen = fungen
        self.satur_pulse = satur_pulse
        self.nucleation_pulse = nucleation_pulse
        self.growth_pulse = growth_pulse
        self.cycles = max(0, int(cycles))
        self.roi = roi
        self.num_images_bg = max(1, int(num_images_bg))
        self.num_images_frame = max(1, int(num_images_frame))
        self.base_outdir = outdir
        os.makedirs(self.base_outdir, exist_ok=True)
        self.campo_corr = float(campo_corr)
        self.resistencia = float(resistencia)
        self.blur_sigma = max(0, int(blur_sigma))
        self.do_resta = bool(do_resta)
        self.mode = mode or "Promedio"
        self.parent_log = parent_log or (lambda msg: None)
        self.background_gain = float(background_gain)
        self.background_offset = float(background_offset)
        self.sequence_mode = str(sequence_mode or "completa").lower()
        self.repetitions = max(1, int(repetitions))

        self.current_outdir = None
        self.current_raw_dir = None

        self._stop = False
        self.background_full = None

    def log(self, message):
        self.parent_log(message)

    def stop(self):
        self._stop = True

    def _should_stop(self):
        return self._stop

    def _capture_stack(self, cantidad):
        imagenes = []
        for _ in range(max(1, cantidad)):
            if self._should_stop():
                break
            im = self.camera.TakeSnapshot()
            if im is not None:
                imagenes.append(np.copy(im))
        if not imagenes:
            raise RuntimeError("No se pudo capturar ninguna imagen.")
        if len(imagenes) == 1:
            resultado = imagenes[0]
        else:
            stack = np.stack(imagenes, axis=0).astype(np.float32)
            modo = str(self.mode).lower()
            if modo.startswith("med"):
                resultado = np.median(stack, axis=0).astype(np.uint16)
            else:
                resultado = np.mean(stack, axis=0).astype(np.uint16)
        if self.blur_sigma > 0:
            resultado = blur_uint16(resultado, sigma=self.blur_sigma)
        return resultado

    def _apply_roi(self, imagen):
        if self.roi is None:
            h, w = imagen.shape
            return imagen, (0, 0, w, h)
        x, y, w, h = self.roi
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))
        w = min(w, imagen.shape[1] - x)
        h = min(h, imagen.shape[0] - y)
        return imagen[y:y + h, x:x + w], (x, y, w, h)

    def _subtract_background(self, roi_img, roi_box):
        if not self.do_resta or self.background_full is None:
            return None
        x, y, w, h = roi_box
        bg_roi = self.background_full[y:y + h, x:x + w]
        image_float = roi_img.astype(np.float32)
        bg_float = bg_roi.astype(np.float32)
        diff = self.background_gain * (image_float - bg_float + self.background_offset)

        min_val = diff.min()
        max_val = diff.max()
        if max_val == min_val:
            return np.zeros_like(roi_img, dtype=np.uint16)

        norm = (diff - min_val) / (max_val - min_val)
        return (norm * 65535).astype(np.uint16)

    def _save_frame(self, full_img, roi_img, roi_box, nombre, indice):
        stem = f"{indice:03d}_{nombre}"
        imsave(os.path.join(self.current_raw_dir, f"{stem}_raw.tif"), full_img)
        resta = self._subtract_background(roi_img, roi_box)
        if resta is not None:
            imsave(os.path.join(self.current_outdir, f"{stem}_resta.tif"), resta)
            return resta
        return roi_img

    def _enviar_pulso(self, etiqueta, configuracion):
        tipo = str(configuracion.get('tipo', 'pulso cuadrado')).lower()
        self.log(f"[SEQ] {etiqueta} — {tipo}")
        if self.fungen is None:
            self.log("[SEQ] Generador no conectado: se omite el pulso.")
            time.sleep(0.05)
            return

        campo = float(configuracion.get('campo', 0.0))
        tiempo = float(configuracion.get('tiempo', 0.0))
        signo = int(configuracion.get('signo', 1))
        amplitud = float(configuracion.get('amplitud', 0.0))
        oscilaciones = int(configuracion.get('oscilaciones', 1))
        triangulos = int(configuracion.get('triangulos', oscilaciones))
        geometria = float(configuracion.get('geometria', 0.5))
        offset = float(configuracion.get('offset', 0.0))

        if tiempo <= 0:
            raise ValueError("El tiempo del pulso debe ser positivo.")
        if self.campo_corr == 0:
            raise ValueError("La constante campo-corriente no puede ser cero.")

        corriente = campo / self.campo_corr
        tension = (corriente * self.resistencia / (10 * 0.95)) / 1000.0
        frecuencia = 1000.0 / tiempo

        if tipo == "pulso oscilatorio":
            forma = sqr_osci_pulse(signo, amplitud, max(1, oscilaciones))
        elif tipo == "pulso triangular":
            forma = triangular_pulse(signo, amplitud, max(1, triangulos), geometria)
        else:
            forma = square_pulse(signo)

        binario = binarize_pulse(forma)

        try:
            self.fungen.write('FREQ %f' % frecuencia)
            time.sleep(0.05)
            self.fungen.write('VOLT:OFFS %f' % offset)
            time.sleep(0.05)
            self.fungen.write('VOLT %f' % tension)
            time.sleep(0.05)
            self.fungen.write('FORM:BORD SWAP')
            time.sleep(0.02)
            self.fungen.write_raw(binario)
            self.fungen.write('FUNC:USER VOLATILE')
            time.sleep(0.02)
            self.fungen.write('FUNC:SHAP USER')
            time.sleep(0.02)
            self.fungen.write('VOLT %f' % tension)
            time.sleep(0.02)
            self.fungen.write('*TRG')
        except Exception as exc:
            raise RuntimeError(f"No se pudo enviar el pulso ({exc})")

    def _prepare_directories(self, rep_idx):
        if self.repetitions == 1:
            outdir = self.base_outdir
        else:
            outdir = os.path.join(self.base_outdir, f"rep_{rep_idx:03d}")
        raw_dir = os.path.join(outdir, "raw")
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        self.current_outdir = outdir
        self.current_raw_dir = raw_dir

    def _emit_progress(self, rep_idx, fraction, message):
        total = max(1, self.repetitions)
        fraction = min(max(fraction, 0.0), 1.0)
        overall = ((rep_idx - 1) + fraction) / total
        pct = int(min(100, overall * 100))
        self.progress.emit(pct, f"Rep {rep_idx}/{total} — {message}")

    def _run_full_sequence(self, rep_idx):
        self._emit_progress(rep_idx, 0.0, "Saturación")
        self._enviar_pulso("Saturación inicial", self.satur_pulse)
        if self._should_stop():
            return

        self._emit_progress(rep_idx, 0.1, "Capturando fondo")
        fondo_full = self._capture_stack(self.num_images_bg)
        self.background_full = fondo_full.copy()
        imsave(os.path.join(self.current_outdir, "fondo_raw.tif"), fondo_full)
        self.bg_ready.emit(fondo_full.copy())
        if self._should_stop():
            return

        self._emit_progress(rep_idx, 0.25, "Nucleación")
        self._enviar_pulso("Pulso de nucleación", self.nucleation_pulse)
        if self._should_stop():
            return

        imagen_nuc = self._capture_stack(self.num_images_frame)
        nuc_roi, nuc_box = self._apply_roi(imagen_nuc)
        preview = self._save_frame(imagen_nuc, nuc_roi, nuc_box, "nucleacion", 0)
        self.image_ready.emit(preview.copy(), 0)
        if self._should_stop():
            return

        for ciclo in range(1, self.cycles + 1):
            if self._should_stop():
                break
            avance = 0.25 + 0.7 * (ciclo / max(1, self.cycles))
            self._emit_progress(rep_idx, min(avance, 0.95), f"Crecimiento {ciclo}/{self.cycles}")
            self._enviar_pulso(f"Pulso de crecimiento {ciclo}", self.growth_pulse)
            if self._should_stop():
                break
            imagen = self._capture_stack(self.num_images_frame)
            roi_img, roi_box = self._apply_roi(imagen)
            preview = self._save_frame(imagen, roi_img, roi_box, f"ciclo_{ciclo:03d}", ciclo)
            self.image_ready.emit(preview.copy(), ciclo)

    def _run_growth_only(self, rep_idx):
        self._emit_progress(rep_idx, 0.0, "Capturando fondo")
        fondo_full = self._capture_stack(self.num_images_bg)
        self.background_full = fondo_full.copy()
        imsave(os.path.join(self.current_outdir, "fondo_raw.tif"), fondo_full)
        self.bg_ready.emit(fondo_full.copy())
        if self._should_stop():
            return

        for ciclo in range(1, self.cycles + 1):
            if self._should_stop():
                break
            avance = 0.05 + 0.9 * (ciclo / max(1, self.cycles))
            self._emit_progress(rep_idx, min(avance, 0.95), f"Crecimiento {ciclo}/{self.cycles}")
            self._enviar_pulso(f"Pulso de crecimiento {ciclo}", self.growth_pulse)
            if self._should_stop():
                break
            imagen = self._capture_stack(self.num_images_frame)
            roi_img, roi_box = self._apply_roi(imagen)
            preview = self._save_frame(imagen, roi_img, roi_box, f"ciclo_{ciclo:03d}", ciclo)
            self.image_ready.emit(preview.copy(), ciclo)

    def run(self):
        try:
            total = self.repetitions
            for rep_idx in range(1, total + 1):
                if self._should_stop():
                    break
                self.log(f"[SEQ] Iniciando repetición {rep_idx}/{total} (modo {self.sequence_mode})")
                self._prepare_directories(rep_idx)
                self.background_full = None

                if self.sequence_mode == "crecer":
                    self._run_growth_only(rep_idx)
                else:
                    self._run_full_sequence(rep_idx)

            if not self._should_stop():
                self._emit_progress(total, 1.0, "Secuencia finalizada")
        except Exception as exc:
            self.log(f"[SEQ][ERROR] {exc}")
        finally:
            self.finished.emit()
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

        # --- Controles manuales de dominios ---
        dom_group = QGroupBox("Saturar muestra y crear dominios")
        dom_layout = QGridLayout(dom_group)

        dom_layout.addWidget(QLabel("Tiempo de saturación [ms]"), 0, 0)
        self.tiempo_saturacion_edit = QLineEdit()
        dom_layout.addWidget(self.tiempo_saturacion_edit, 0, 1)

        dom_layout.addWidget(QLabel("Campo de saturación [Oe]"), 0, 2)
        self.campo_saturacion_edit = QLineEdit()
        dom_layout.addWidget(self.campo_saturacion_edit, 0, 3)

        dom_layout.addWidget(QLabel("Tiempo de dominio [ms]"), 1, 0)
        self.tiempo_dominio_edit = QLineEdit()
        dom_layout.addWidget(self.tiempo_dominio_edit, 1, 1)

        dom_layout.addWidget(QLabel("Campo de dominio [Oe]"), 1, 2)
        self.campo_dominio_edit = QLineEdit()
        dom_layout.addWidget(self.campo_dominio_edit, 1, 3)

        dom_layout.addWidget(QLabel("Constante campo-corriente [G/mA]"), 2, 0)
        self.campo_corr_edit = QLineEdit("0.3956")
        dom_layout.addWidget(self.campo_corr_edit, 2, 1)

        dom_layout.addWidget(QLabel("Resistencia [Ω]"), 2, 2)
        self.resistencia_edit = QLineEdit("103")
        dom_layout.addWidget(self.resistencia_edit, 2, 3)

        dom_layout.addWidget(QLabel("Signo del dominio:"), 3, 0)
        self.radio_signo_pos_dom = QRadioButton("Positivo")
        self.radio_signo_neg_dom = QRadioButton("Negativo")
        self.radio_signo_pos_dom.setChecked(True)

        self.signo_group_dom = QButtonGroup()
        self.signo_group_dom.addButton(self.radio_signo_pos_dom)
        self.signo_group_dom.addButton(self.radio_signo_neg_dom)

        signo_layout_dom = QHBoxLayout()
        signo_layout_dom.addWidget(self.radio_signo_pos_dom)
        signo_layout_dom.addWidget(self.radio_signo_neg_dom)
        dom_layout.addLayout(signo_layout_dom, 3, 1, 1, 3)

        self.saturate_dom_button = QPushButton("Saturar")
        self.create_dom_button = QPushButton("Crear dominios")
        botones_layout = QHBoxLayout()
        botones_layout.addWidget(self.saturate_dom_button)
        botones_layout.addWidget(self.create_dom_button)
        dom_layout.addLayout(botones_layout, 4, 0, 1, 4)

        main_layout.addWidget(dom_group)

        # --- Sección de secuencias ---
        sequence_section = self._build_sequence_section()
        main_layout.addWidget(sequence_section)

        main_layout.addStretch()

        self.control_tab.setLayout(main_layout)

        # Deshabilitar acciones dependientes del generador hasta conectar
        self.gen_buttons = [self.saturate_dom_button, self.create_dom_button]
        if hasattr(self, "seq_run_btn"):
            self.gen_buttons.append(self.seq_run_btn)
        for btn in self.gen_buttons:
            btn.setEnabled(False)

        # Conexiones
        self.saturate_dom_button.clicked.connect(self.saturate_dom)
        self.create_dom_button.clicked.connect(self.create_dom)


    def _make_sequence_pulse_group(self, title):
        group = QGroupBox(title)
        layout = QGridLayout(group)

        campo_label = QLabel("Campo [Oe]:")
        campo_spin = QDoubleSpinBox()
        campo_spin.setRange(0.0, 5000.0)
        campo_spin.setDecimals(2)
        campo_spin.setSingleStep(10.0)
        campo_spin.setValue(100.0)
        layout.addWidget(campo_label, 0, 0)
        layout.addWidget(campo_spin, 0, 1)

        tiempo_label = QLabel("Tiempo [ms]:")
        tiempo_spin = QDoubleSpinBox()
        tiempo_spin.setRange(0.1, 10000.0)
        tiempo_spin.setDecimals(2)
        tiempo_spin.setSingleStep(1.0)
        tiempo_spin.setValue(10.0)
        layout.addWidget(tiempo_label, 0, 2)
        layout.addWidget(tiempo_spin, 0, 3)

        tipo_label = QLabel("Tipo de pulso:")
        tipo_combo = QComboBox()
        tipo_combo.addItems(["Pulso cuadrado", "Pulso oscilatorio", "Pulso triangular"])
        layout.addWidget(tipo_label, 1, 0)
        layout.addWidget(tipo_combo, 1, 1)

        signo_label = QLabel("Signo:")
        signo_combo = QComboBox()
        signo_combo.addItem("Positivo", 1)
        signo_combo.addItem("Negativo", -1)
        layout.addWidget(signo_label, 1, 2)
        layout.addWidget(signo_combo, 1, 3)

        amp_label = QLabel("Amplitud oscilación [%]:")
        amp_spin = QDoubleSpinBox()
        amp_spin.setRange(0.0, 50.0)
        amp_spin.setDecimals(1)
        amp_spin.setSingleStep(1.0)
        amp_spin.setValue(0.0)
        layout.addWidget(amp_label, 2, 0)
        layout.addWidget(amp_spin, 2, 1)

        osc_label = QLabel("Oscilaciones:")
        osc_spin = QSpinBox()
        osc_spin.setRange(1, 100)
        osc_spin.setValue(1)
        layout.addWidget(osc_label, 2, 2)
        layout.addWidget(osc_spin, 2, 3)

        geom_label = QLabel("Factor geometría:")
        geom_spin = QDoubleSpinBox()
        geom_spin.setRange(0.0, 1.0)
        geom_spin.setDecimals(2)
        geom_spin.setSingleStep(0.05)
        geom_spin.setValue(0.5)
        layout.addWidget(geom_label, 3, 0)
        layout.addWidget(geom_spin, 3, 1)

        def toggle_extra(text):
            text_lower = text.lower()
            is_osci = "oscilatorio" in text_lower
            is_tri = "triangular" in text_lower

            amp_label.setText("Amplitud oscilación [%]:" if is_osci else "Amplitud triangular [%]:")
            amp_label.setVisible(is_osci or is_tri)
            amp_spin.setVisible(is_osci or is_tri)

            osc_label.setText("Oscilaciones:" if is_osci else "Triángulos:")
            osc_label.setVisible(is_osci or is_tri)
            osc_spin.setVisible(is_osci or is_tri)

            geom_label.setVisible(is_tri)
            geom_spin.setVisible(is_tri)

        tipo_combo.currentTextChanged.connect(toggle_extra)
        toggle_extra(tipo_combo.currentText())

        controls = {
            "campo": campo_spin,
            "tiempo": tiempo_spin,
            "tipo": tipo_combo,
            "signo": signo_combo,
            "amplitud": amp_spin,
            "oscilaciones": osc_spin,
            "geometria": geom_spin,
        }
        return group, controls

    def _build_sequence_section(self):
        container = QGroupBox("Secuencia simple")
        layout = QHBoxLayout(container)

        controls_box = QGroupBox("Pulsos")
        controls_layout = QVBoxLayout(controls_box)
        self.seq_controls = {}

        for key, title in [("saturacion", "Pulso de saturación"),
                            ("nucleacion", "Pulso de nucleación"),
                            ("crecimiento", "Pulso de crecimiento")]:
            group, controls = self._make_sequence_pulse_group(title)
            controls_layout.addWidget(group)
            self.seq_controls[key] = controls

        cycles_layout = QHBoxLayout()
        cycles_layout.addWidget(QLabel("Ciclos (pulso + foto):"))
        self.seq_cycle_spin = QSpinBox()
        self.seq_cycle_spin.setRange(1, 1000)
        self.seq_cycle_spin.setValue(5)
        cycles_layout.addWidget(self.seq_cycle_spin)
        controls_layout.addLayout(cycles_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Modo:"))
        self.seq_mode_combo = QComboBox()
        self.seq_mode_combo.addItem("Completa", "completa")
        self.seq_mode_combo.addItem("Crecer", "crecer")
        mode_layout.addWidget(self.seq_mode_combo)

        mode_layout.addWidget(QLabel("Repeticiones:"))
        self.seq_repeat_spin = QSpinBox()
        self.seq_repeat_spin.setRange(1, 100)
        self.seq_repeat_spin.setValue(1)
        mode_layout.addWidget(self.seq_repeat_spin)
        controls_layout.addLayout(mode_layout)

        buttons_layout = QHBoxLayout()
        self.seq_run_btn = QPushButton("Iniciar secuencia")
        self.seq_stop_btn = QPushButton("Detener")
        self.seq_stop_btn.setEnabled(False)
        self.seq_run_btn.clicked.connect(self._seq_run)
        self.seq_stop_btn.clicked.connect(self._seq_stop)
        buttons_layout.addWidget(self.seq_run_btn)
        buttons_layout.addWidget(self.seq_stop_btn)
        controls_layout.addLayout(buttons_layout)

        self.seq_progress = QLabel("Listo")
        controls_layout.addWidget(self.seq_progress)

        layout.addWidget(controls_box, 2)

        preview_group = QGroupBox("Última imagen")
        preview_layout = QVBoxLayout(preview_group)
        self.seq_preview = QLabel("\n\n(Se mostrará aquí)")
        self.seq_preview.setAlignment(Qt.AlignCenter)
        self.seq_preview.setMinimumSize(260, 200)
        self.seq_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self.seq_preview)

        layout.addWidget(preview_group, 3)

        return container

    def _sequence_signature(self):
        if not hasattr(self, 'seq_controls'):
            return "secuencia"
        partes = []
        for key, prefijo in [("saturacion", "Sat"),
                              ("nucleacion", "Nuc"),
                              ("crecimiento", "Grow")]:
            controls = self.seq_controls.get(key)
            if not controls:
                continue
            campo = controls["campo"].value()
            tiempo = controls["tiempo"].value()
            tipo = controls["tipo"].currentText().lower()
            amplitud = controls["amplitud"].value()
            if campo > 0 and tiempo > 0:
                extra = ""
                if "oscilatorio" in tipo or "triangular" in tipo:
                    extra = f"_Amp{amplitud:g}"
                partes.append(f"{prefijo}{campo:g}Oe_{tiempo:g}ms{extra}")
        return "_".join(partes) if partes else "secuencia"

    def _pulse_config_from_controls(self, key):
        controls = self.seq_controls.get(key, {})
        config = {
            "campo": controls.get("campo").value() if controls.get("campo") else 0.0,
            "tiempo": controls.get("tiempo").value() if controls.get("tiempo") else 0.0,
            "tipo": controls.get("tipo").currentText() if controls.get("tipo") else "Pulso cuadrado",
            "signo": controls.get("signo").currentData() if controls.get("signo") else 1,
            "amplitud": controls.get("amplitud").value() if controls.get("amplitud") else 0.0,
            "oscilaciones": controls.get("oscilaciones").value() if controls.get("oscilaciones") else 1,
            "triangulos": controls.get("oscilaciones").value() if controls.get("oscilaciones") else 1,
            "geometria": controls.get("geometria").value() if controls.get("geometria") else 0.5,
        }
        return config

    def iniciar_conexion(self):
        """
        Intenta conectar con los equipos vía PyVISA.
        Según los resultados, habilita los botones correspondientes.
        """
        rm = visa.ResourceManager()
        self.fungen = None

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
                except Exception as e:
                    print(f"{resource}: No se pudo indentificar ({e})")

            self.fungen = rm.open_resource("GPIB0::10::INSTR")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo escanear recursos VISA:\n{e}")
            return

        if self.fungen is None:
            QMessageBox.warning(self, "Conexión fallida", "❌ No se detectó el generador de funciones.")
            return

        try:
            self.fungen.write("OUTP:LOAD INF")    # impedancia de salida: High Z
            self.fungen.write("BM:SOUR INT")      # fuente de ráfaga interna
            self.fungen.write("BM:NCYC 1")        # 1 ciclo por ráfaga
            self.fungen.write("BM:PHASe 0")       # fase inicial 0°
            self.fungen.write("BM:STAT ON")       # activar modo burst
            self.fungen.write("TRIG:SOUR BUS")    # trigger por software
        except Exception as e:
            QMessageBox.warning(self, "Generador", f"⚠️ No se pudo configurar el generador:\n{e}")

        for btn in getattr(self, "gen_buttons", []):
            btn.setEnabled(True)

        QMessageBox.information(self, "Conexión completada", "✅ Generador conectado y configurado.")

    def _parse_int_or(self, text, default=1):
        try:
            return max(1, int(float(text)))
        except Exception:
            return default
    
    def _apply_burst_cycles_from_ui(self):
        if not hasattr(self, "fungen") or self.fungen is None:
            return
        nro_widget = getattr(self, "nro_ciclo_edit", None)
        if nro_widget is None:
            return
        ncyc = self._parse_int_or(nro_widget.text(), 1)
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
            cycles = int(self.seq_cycle_spin.value())
        except Exception:
            cycles = 1

        sequence_mode = "completa"
        repetitions = 1
        if hasattr(self, "seq_mode_combo"):
            sequence_mode = str(self.seq_mode_combo.currentData() or "completa")
        if hasattr(self, "seq_repeat_spin"):
            try:
                repetitions = max(1, int(self.seq_repeat_spin.value()))
            except Exception:
                repetitions = 1

        try:
            campo_corr = float(self.campo_corr_edit.text())
            resistencia = float(self.resistencia_edit.text())
        except Exception:
            QMessageBox.warning(self, "Constantes faltantes", "Ingresá la constante campo-corriente y la resistencia antes de iniciar la secuencia.")
            return

        satur_config = self._pulse_config_from_controls("saturacion")
        nuc_config = self._pulse_config_from_controls("nucleacion")
        grow_config = self._pulse_config_from_controls("crecimiento")

        # Carpeta destino en work_dir (o cwd) con firma + timestamp
        sign = self._sequence_signature()
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = self.work_dir if self.work_dir else os.getcwd()
        modo_tag = "CRECER" if sequence_mode == "crecer" else "FULL"
        outdir = os.path.join(base_dir, f"SEQ_{sign}_{modo_tag}_{ts}")

        roi = self._roi_tuple_or_none()

        blur_sigma = getattr(self, 'blur_strength', 0)
        num_images_widget = getattr(self, 'num_images_spinbox', None)
        if num_images_widget is not None:
            num_images = max(1, int(num_images_widget.value()))
        else:
            num_images = 1

        # Parámetros de resta de fondo y blur actuales (si usás otros nombres, ajustá aquí)
        capture_mode = getattr(self, 'capture_mode', 'Promedio')

        # Preparar Worker + hilo
        self.seq_thread = QThread()
        self.seq_worker = SequenceWorker(
            camera=self.camera,
            fungen=getattr(self, 'fungen', None),
            satur_pulse=satur_config,
            nucleation_pulse=nuc_config,
            growth_pulse=grow_config,
            cycles=cycles,
            roi=roi,
            num_images_bg=num_images,
            num_images_frame=num_images,
            outdir=outdir,
            campo_corr=campo_corr,
            resistencia=resistencia,
            blur_sigma=blur_sigma,
            do_resta=True,
            mode=capture_mode,
            parent_log=self._seq_threadsafe_log,
            background_gain=getattr(self, 'background_gain', 1.0),
            background_offset=getattr(self, 'background_offset', 0.0),
            sequence_mode=sequence_mode,
            repetitions=repetitions,
        )
        self.seq_worker.moveToThread(self.seq_thread)

        if getattr(self, 'fungen', None) is None:
            self.log_message("[SEQ] Generador no conectado: los pulsos se omitirán.")

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
        self.log_message(f"[SEQ] Iniciada — modo {sequence_mode}, repeticiones {repetitions}")
    
    
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
            pix = self._to_qpixmap(bg_img)
            self.seq_preview.setPixmap(pix.scaled(
                self.seq_preview.width(), self.seq_preview.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
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
        thread = getattr(self, 'seq_thread', None)
        try:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(2000)
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
        config = self._pulse_config_from_controls("crecimiento")

        try:
            campo_corr = float(self.campo_corr_edit.text())
            resistencia = float(self.resistencia_edit.text())
        except Exception:
            self.log_message("[GROW] Definí la constante campo-corriente y la resistencia.")
            return

        if campo_corr == 0:
            self.log_message("[GROW] La constante campo-corriente debe ser distinta de cero.")
            return

        if not getattr(self, 'fungen', None):
            self.log_message("[GROW] Generador no conectado.")
            return

        try:
            campo = float(config.get("campo", 0.0))
            tiempo = float(config.get("tiempo", 0.0))
            signo = int(config.get("signo", 1))
            tipo = str(config.get("tipo", "Pulso cuadrado")).lower()
            amplitud = float(config.get("amplitud", 0.0))
            oscilaciones = int(config.get("oscilaciones", 1))
            triangulos = int(config.get("triangulos", oscilaciones))
            geometria = float(config.get("geometria", 0.5))
        except Exception:
            self.log_message("[GROW] Parámetros inválidos para el pulso de crecimiento.")
            return

        if tiempo <= 0:
            self.log_message("[GROW] El tiempo del pulso debe ser positivo.")
            return

        corriente = campo / campo_corr
        tension = (corriente * resistencia / (10 * 0.95)) / 1000.0
        frecuencia = 1000.0 / tiempo

        if "oscilatorio" in tipo:
            forma = sqr_osci_pulse(signo, amplitud, oscilaciones)
        elif "triangular" in tipo:
            forma = triangular_pulse(signo, amplitud, triangulos, geometria)
        else:
            forma = square_pulse(signo)

        binario = binarize_pulse(forma)

        try:
            if self.fungen.query("BM:STAT?") == 0:
                self.log_message("[GROW] El modo ráfaga está deshabilitado en el generador.")
                return
        except Exception:
            self.log_message("[GROW] No se pudo consultar el modo ráfaga del generador.")

        try:
            self.fungen.write('FREQ %f' % frecuencia)
            time.sleep(0.05)
            self.fungen.write('VOLT:OFFS 0')
            time.sleep(0.05)
            self.fungen.write('VOLT %f' % tension)
            time.sleep(0.05)
            self.fungen.write('FORM:BORD SWAP')
            time.sleep(0.02)
            self.fungen.write_raw(binario)
            self.fungen.write('FUNC:USER VOLATILE')
            time.sleep(0.02)
            self.fungen.write('FUNC:SHAP USER')
            time.sleep(0.02)
            self.fungen.write('VOLT %f' % tension)
            time.sleep(0.02)
            self.fungen.write('*TRG')
        except Exception as exc:
            self.log_message(f"[GROW][ERROR] No se pudo enviar el pulso: {exc}")
            
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
        
        pulso = square_pulse(signo)        
        binario = binarize_pulse(pulso)
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
        
        pulso = square_pulse(signo)        
        binario = binarize_pulse(pulso)
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
        
        combo = getattr(self, "combo", None)
        if combo is not None:
            combo.clear()
            for nombre_config, valores in datos.items():
                combo.addItem(nombre_config, valores)


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


    