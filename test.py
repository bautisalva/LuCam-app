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

Dependencies:
    - PyQt6
    - NumPy
    - OpenCV (cv2)
    - lucam (optional)

Author: Tomás Rodriguez Bouhier, Bautista Salvatierra Pérez
Repository: https://github.com/bautisalva/LuCam-app
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import threading
import json
import datetime
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.io import imsave
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, 
                             QSpinBox, QComboBox, QFileDialog,
                             QGroupBox, QTabWidget, QGridLayout,QPlainTextEdit)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread


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

def rescale_to_full_uint16(image):
    """
    Rescala una imagen uint16 al rango completo [0–65535], usando percentiles
    para evitar que outliers (líneas oscuras, picos) arruinen el contraste.
    """
    p_low, p_high = np.percentile(image, (1, 99))  # 1% y 99% para proteger extremos
    if p_high == p_low:
        return np.zeros_like(image, dtype=np.uint16)
    scaled = ((image - p_low) / (p_high - p_low) * 65535).clip(0, 65535).astype(np.uint16)
    return scaled

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
    def __init__(self):
        super().__init__()
    
        # Try initializing Lucam camera; fallback to simulation
        try:
            self.camera = Lucam()
            self.simulation = False
            print("[INFO] Cámara Lucam inicializada correctamente.")
        except Exception as e:
            print(f"[WARNING] No se pudo inicializar Lucam. Se usará SimulatedCamera. Error: {e}")
            self.camera = SimulatedCamera()
            self.simulation = True
        
        # Configuración del formato del frame
        frameformat, fps = self.camera.GetFormat()
        frameformat.pixelFormat = API.LUCAM_PF_16
        self.camera.SetFormat(frameformat, fps)
    
        # Logging system
        self.log_file_path = os.path.join(os.getcwd(), "log.txt")
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
        
        now = datetime.datetime.now()
        start_message = f"=== Se inició la app el día {now.strftime('%d/%m/%Y')} a las {now.strftime('%H:%M:%S')} ==="
        self.log_file.write(start_message + "\n")
        self.log_file.flush()
    
        # Start preview worker thread
        self.preview_worker = PreviewWorker(self.camera)
        self.preview_thread = QThread()
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_worker.new_frame.connect(self.display_preview_image)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_thread.start()
    
        # Initialize GUI widgets and internal state variables
        self.preview_label_preview = QLabel("Preview en vivo")
        self.preview_label_preview.setFixedSize(640, 480)
    
        self.preview_label_capture = QLabel("Preview captura")
        self.preview_label_capture.setFixedSize(640, 480)
    
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
            "gamma": (1, 50, 10),
            "exposure": (1, 500, 10.0),
            "gain": (0, 10, 1.0),
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
    
        # Add window
        self.tabs.addTab(self.preview_tab, "Preview")
        self.tabs.addTab(self.capture_tab, "Captura")
    
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
            slider.setMinimum(int(min_val * 10))
            slider.setMaximum(int(max_val * 10))
            slider.setValue(int(default * 10))
            slider.valueChanged.connect(lambda value, p=prop: self.update_property(p, value / 10))
    
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
    
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
    
        self.apply_default_slider_values_to_camera()
    
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
    
        controls_layout.addStretch()
    
        layout.addLayout(controls_layout)
    
        self.capture_tab.setLayout(layout)

    def apply_default_slider_values_to_camera(self):
            """
            Applies the default values defined in self.properties
            to the connected camera using set_properties().
            """
            for prop, (min_val, max_val, default) in self.properties.items():
                if self.camera:
                    self.camera.set_properties(**{prop: default})
    
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
        self.sliders[prop].blockSignals(True)
        self.sliders[prop].setValue(int(value * 10))
        self.sliders[prop].blockSignals(False)
        self.inputs[prop].setText(f"{value:.1f}")
        self.log_message(f"Se actualizó '{prop}' a {value:.1f}")
        
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
            self.sliders[prop].setValue(int(value * 10))
            self.update_property(prop, value)
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
        
    
    def save_image_automatically(self, image, tipo):
        """
        Saves an image to disk using timestamped filename in working directory.

        Parameters:
            image (np.ndarray): image to save.
            tipo (str): label prefix (e.g., 'fondo', 'resta').
        """
        if not self.work_dir:
            self.log_message("[ERROR] No se definió directorio de trabajo para guardar imagen.")
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tipo}_{timestamp}.tif"
        full_path = os.path.join(self.work_dir, filename)
        try:
            image_to_save = rescale_to_full_uint16(image)
            imsave(full_path, image_to_save)
            self.log_message(f"Imagen guardada automáticamente en: {full_path}")
        except Exception as e:
            self.log_message(f"[ERROR] No se pudo guardar la imagen en: {full_path}. Detalle: {e}")
    
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
        if self.auto_save:
            self.save_image_automatically(image, "fondo")
    
    def toggle_background(self, text):
        """
        Enables or disables background subtraction mode.

        Parameters:
            text (str): "Sí" to enable, "No" to disable.
        """
        self.background_enabled = (text == "Sí")
        estado = "activada" if self.background_enabled else "desactivada"
        self.log_message(f"Restar fondo {estado}")
    
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
        if (self.background_enabled and
                self.background_image is not None and
                self.background_image.shape == image.shape):
    
            a = self.background_gain
            b = self.background_offset
    
            image_float = image.astype(np.float32)
            background_float = self.background_image.astype(np.float32)
    
            diff = a * (image_float - background_float + b)
            diff_centered = diff + 32768
            result = np.clip(diff_centered, 0, 65535).astype(np.uint16)
            tipo_guardado = "resta"
        else:
            result = image.copy()
            tipo_guardado = "normal"
    
        self.captured_image = result
    
        # ⚠️ MUY IMPORTANTE: mostrar la imagen procesada de 16 bits
        self.display_captured_image_in_tab(result)
    
        self.save_button.setEnabled(True)
        self.toggle_preview_button.setEnabled(True)
        self.log_message("Imagen capturada y mostrada en pestaña 'Captura'.")
    
        if self.auto_save:
            self.save_image_automatically(result, tipo_guardado)
            # Guardar RAW también
            raw_folder = os.path.join(self.work_dir, "raw")
            os.makedirs(raw_folder, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cruda_{timestamp}.tif"
            raw_path = os.path.join(raw_folder, filename)
            imsave(raw_path, image)
            self.log_message(f"Imagen cruda guardada en: {raw_path}")

    def display_captured_image_in_tab(self, image, scale_factor=1):
        if len(image.shape) == 3:
            image = (rgb2gray(image) * 65535).astype(np.uint16)
    
        height, width = image.shape
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
    
        resized_image = resize(image, (new_height, new_width), preserve_range=True).astype(np.uint16)
        image_8bit = to_8bit_for_preview(resized_image)
    
        # ✅ Aseguramos que el array sea contiguo en memoria
        image_8bit = np.ascontiguousarray(image_8bit)
    
        bytes_per_line = image_8bit.shape[1]
        qimage = QImage(image_8bit.data, image_8bit.shape[1], image_8bit.shape[0], bytes_per_line, QImage.Format.Format_Grayscale8)
        self.preview_label_capture.setPixmap(QPixmap.fromImage(qimage))



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
            "background_offset": self.background_offset
        }
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Parámetros de Captura", "", "JSON (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(params, f, indent=4)
                self.log_message(f"Parámetros de Captura guardados en {file_path}")
            except Exception as e:
                self.log_message(f"[ERROR] No se pudieron guardar parámetros de Captura: {e}")
                
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
                    self.sliders[prop].setValue(int(value * 10))
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



    