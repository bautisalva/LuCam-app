import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
from skimage.transform import resize
from skimage.io import imsave,imread
from skimage.color import rgb2gray
from utils import to_8bit_for_preview
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, 
                             QSpinBox, QComboBox, QFileDialog,
                             QGroupBox, QTabWidget, QGridLayout,QPlainTextEdit)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QRect


class PreviewTab(QWidget):
    def __init__(self, camera, log_message, available_fps, properties, simulation=False):
        super().__init__()
        self.camera = camera
        self.log_message = log_message
        self.available_fps = available_fps
        self.properties = properties
        self.simulation = simulation
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()

        # Lado izquierdo: preview y consola
        left_layout = QVBoxLayout()
        self.preview_label = QLabel("Preview en vivo")
        self.preview_label.setFixedSize(960, 720)
        left_layout.addWidget(self.preview_label)
        
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        left_layout.addWidget(self.console)
        layout.addLayout(left_layout)

        # Controles
        controls_layout = QVBoxLayout()
        self.sliders = {}
        self.inputs = {}
        
        # Grupo FPS
        fps_group = QGroupBox("FPS")
        fps_layout = QHBoxLayout()
        self.fps_selector = QComboBox()
        for fps in self.available_fps:
            self.fps_selector.addItem(f"{fps:.2f}")
        self.fps_selector.currentTextChanged.connect(self.change_fps)
        fps_layout.addWidget(QLabel("Frames por segundo:"))
        fps_layout.addWidget(self.fps_selector)
        fps_group.setLayout(fps_layout)
        controls_layout.addWidget(fps_group)

        # Propiedades
        for prop, (min_val, max_val, default) in self.properties.items():
            group = QGroupBox(prop.capitalize())
            group_layout = QHBoxLayout()
            
            label = QLabel(f"{default}")
            slider = QSlider(Qt.Horizontal)
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

        # Botones
        self.save_button = QPushButton("Guardar Parámetros de Preview")
        self.save_button.clicked.connect(self.save_preview_parameters)
        controls_layout.addWidget(self.save_button)
        
        self.load_button = QPushButton("Cargar Parámetros")
        self.load_button.clicked.connect(self.load_parameters)
        controls_layout.addWidget(self.load_button)
        
        self.refresh_button = QPushButton("Refrescar desde Cámara")
        self.refresh_button.clicked.connect(self.apply_real_camera_values)
        controls_layout.addWidget(self.refresh_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        self.setLayout(layout)
        
        self.apply_real_camera_values()

    def change_fps(self, fps_text):
        try:
            fps = float(fps_text)
            frameformat, _ = self.camera.GetFormat()
            self.camera.SetFormat(frameformat, fps)
            self.log_message(f"FPS cambiado a {fps:.2f}")
        except Exception as e:
            self.log_message(f"[ERROR] No se pudo cambiar FPS: {e}")

    def update_property(self, prop, value):
        self.camera.set_properties(**{prop: value})
        self.sliders[prop].blockSignals(True)
        self.sliders[prop].setValue(int(value * 100))
        self.sliders[prop].blockSignals(False)
        self.inputs[prop].setText(f"{value:.2f}")
        self.log_message(f"Propiedad '{prop}' actualizada a {value:.2f}")

    def set_property_from_input(self, prop, field):
        try:
            value = float(field.text())
            self.update_property(prop, value)
        except ValueError:
            field.setText(f"{self.sliders[prop].value() / 100:.2f}")

    def apply_real_camera_values(self):
        for prop in self.properties:
            try:
                if self.simulation:
                    value = self.properties[prop][2]
                else:
                    value, _ = self.camera.GetProperty(prop)
                self.sliders[prop].setValue(int(value * 100))
                self.inputs[prop].setText(f"{value:.2f}")
            except Exception as e:
                self.log_message(f"[ERROR] Error leyendo propiedad '{prop}': {e}")

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
    
        self.preview_label.setPixmap(QPixmap.fromImage(qimage))


