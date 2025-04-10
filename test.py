import sys
import numpy as np
import cv2
import threading
import json
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, 
                             QSpinBox, QComboBox, QFileDialog, QGroupBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject

from lucam import Lucam

#------------------------------------------------------------------------------
class Worker(QObject):
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
                images.append(image)

            if not images:
                print("[ERROR] No se capturó ninguna imagen válida.")
                return

            # Convertimos a float para el promedio si es necesario
            if self.mode == "Promedio":
                stack = np.stack(images).astype(np.float32)
                result_image = np.mean(stack, axis=0).astype(np.uint8)
            elif self.mode == "Mediana":
                stack = np.stack(images).astype(np.uint8)
                result_image = np.median(stack, axis=0).astype(np.uint8)
            else:
                print(f"[WARNING] Modo desconocido: {self.mode}, se usa la primera imagen.")
                result_image = images[0]

            if self.blur_strength > 0:
                k = self.blur_strength * 2 + 1
                result_image = cv2.GaussianBlur(result_image, (k, k), 0)
                
            self.image_captured.emit(result_image)

        except Exception as e:
            print(f"[ERROR] Falló la captura con Lucam: {e}")

#------------------------------------------------------------------------------
class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.camera = Lucam()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.preview_mode = True
        self.timer.start(300)
        self.capture_mode = "Promedio"
        self.captured_image = None
        self.blur_strength = 0
        self.background_image = None
        self.background_enabled = True
        self.properties = {
            "brightness": (0, 100, 10.0),
            "contrast": (0, 10, 1.0),
            "saturation": (0, 100, 10.0),
            "hue": (-180, 180, 0.0),
            "gamma": (1, 50, 10),
            "exposure": (1, 100, 10.0),
            "gain": (0, 10, 1.0)
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Lumenera Camera Control")
        self.setGeometry(100, 100, 1000, 600)
    
        self.preview_label = QLabel("Preview en vivo")
        self.preview_label.setFixedSize(640, 480)
    
        controls_layout = QVBoxLayout()
    
        self.capture_button = QPushButton("Capturar Imagen")
        self.capture_button.clicked.connect(self.capture_image)
    
        self.capture_background_button = QPushButton("Capturar Fondo")
        self.capture_background_button.clicked.connect(self.capture_background)
    
        self.toggle_background_selector = QComboBox()
        self.toggle_background_selector.addItems(["Sin Fondo", "Con Fondo"])
        self.toggle_background_selector.currentTextChanged.connect(self.toggle_background)
    
        self.save_button = QPushButton("Guardar Imagen")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
    
        self.toggle_preview_button = QPushButton("Volver a Preview")
        self.toggle_preview_button.clicked.connect(self.start_preview)
        self.toggle_preview_button.setEnabled(False)
    
        self.save_settings_button = QPushButton("Guardar Parámetros")
        self.save_settings_button.clicked.connect(self.save_parameters)
    
        self.load_settings_button = QPushButton("Cargar Parámetros")
        self.load_settings_button.clicked.connect(self.load_parameters)
    
        # Capture settings group
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
    
        # Blur control
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
    
        # Camera controls group
        camera_controls_box = QGroupBox("Controles de la Cámara")
        camera_controls_layout = QVBoxLayout()
    
        self.sliders = {}
        self.inputs = {}
        for prop, (min_val, max_val, default) in self.properties.items():
            label = QLabel(f"{prop.capitalize()}: {default}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(min_val * 10))
            slider.setMaximum(int(max_val * 10))
            slider.setValue(int(default * 10))
            slider.valueChanged.connect(lambda value, p=prop: self.update_property(p, value / 10))
    
            input_field = QLineEdit(str(default))
            input_field.setFixedWidth(50)
            input_field.editingFinished.connect(lambda p=prop, field=input_field: self.set_property_from_input(p, field))
    
            hbox = QHBoxLayout()
            hbox.addWidget(label)
            hbox.addWidget(slider)
            hbox.addWidget(input_field)
    
            camera_controls_layout.addLayout(hbox)
            self.sliders[prop] = slider
            self.inputs[prop] = input_field
    
        camera_controls_box.setLayout(camera_controls_layout)
    
        # Add all to controls layout
        controls_layout.addWidget(capture_settings_box)
        controls_layout.addWidget(self.capture_button)
        controls_layout.addWidget(self.capture_background_button)
        controls_layout.addWidget(self.toggle_background_selector)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.toggle_preview_button)
        controls_layout.addWidget(self.save_settings_button)
        controls_layout.addWidget(self.load_settings_button)
        controls_layout.addWidget(camera_controls_box)
        controls_layout.addStretch()
    
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.preview_label)
        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)


    def update_property(self, prop, value):
        if self.camera:
            self.camera.set_properties(**{prop: value})
        self.inputs[prop].setText(f"{value:.1f}")

    def set_property_from_input(self, prop, field):
        try:
            value = float(field.text())
            self.sliders[prop].setValue(int(value * 10))
            self.update_property(prop, value)
        except ValueError:
            field.setText(f"{self.sliders[prop].value() / 10:.1f}")
            
    def update_blur(self, value):
        self.blur_strength = value  # <-- no lo transformes acá
        self.blur_label.setText(f"Blur: {value}")
        self.blur_input.setText(str(value))

    def set_blur_from_input(self):
        try:
            value = int(self.blur_input.text())
            if 0 <= value <= 10:
                self.blur_slider.setValue(value)
            else:
                self.blur_input.setText(str(self.blur_slider.value()))
        except ValueError:
            self.blur_input.setText(str(self.blur_slider.value()))

    def change_capture_mode(self, mode):
        self.capture_mode = mode

    def capture_background(self):
        self.preview_mode = False
        self.timer.stop()
    
        num_images = self.num_images_spinbox.value()
        mode = self.capture_mode_selector.currentText()
        blur_strength = self.blur_strength
    
        self.background_worker = Worker(self.camera, num_images, mode, blur_strength)
        self.background_worker.image_captured.connect(self.set_background_image)
    
        threading.Thread(target=self.background_worker.run, daemon=True).start()


    def set_background_image(self, image):
        self.background_image = image
        self.display_image(image)
        print("Fondo capturado.")

    def toggle_background(self, text):
        self.background_enabled = (text == "Fondo Activado")
        if self.captured_image is not None:
            self.display_captured_image(self.captured_image)

    def capture_image(self):
        num_images = self.num_images_spinbox.value()
        mode = self.capture_mode_selector.currentText()  # importante usar este
        self.preview_mode = False
        self.timer.stop()
    
        self.worker = Worker(self.camera, num_images, mode, self.blur_strength)
        self.worker.image_captured.connect(self.display_captured_image)
        threading.Thread(target=self.worker.run, daemon=True).start()
        
    def display_captured_image(self, image):
        if (self.background_enabled 
                and self.background_image is not None 
                and self.background_image.shape == image.shape):
        
            image_int16 = image.astype(np.int16)
            background_int16 = self.background_image.astype(np.int16)
            diff = image_int16 - background_int16
            diff_shifted = diff + 255
            diff_normalized = np.clip((diff_shifted / 2), 0, 255).astype(np.uint8)
            result = diff_normalized
        else:
            result = image.copy()

        self.captured_image = result
        self.display_image(result)
        self.save_button.setEnabled(True)
        self.toggle_preview_button.setEnabled(True)

        self.captured_image = result
        self.display_image(result)
        self.save_button.setEnabled(True)
        self.toggle_preview_button.setEnabled(True)

    def save_image(self):
        if self.captured_image is None:
            return

        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix("tif")
        file_path, _ = file_dialog.getSaveFileName(self, "Guardar Imagen", "", "TIFF (*.tif)")

        if file_path:
            cv2.imwrite(file_path, self.captured_image)
            print(f"Imagen guardada en {file_path}")
            
    def save_parameters(self):
        params = {prop: self.sliders[prop].value() / 10 for prop in self.properties}
        params['blur'] = self.blur_slider.value()
        params['num_images'] = self.num_images_spinbox.value()
        params['capture_mode'] = self.capture_mode_selector.currentText()

        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Parámetros", "", "JSON (*.json)")

        if file_path:
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"Parámetros guardados en {file_path}")

    def load_parameters(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Cargar Parámetros", "", "JSON (*.json)")

        if file_path:
            with open(file_path, 'r') as f:
                params = json.load(f)

            for prop in self.properties:
                value = params.get(prop, None)
                if value is not None:
                    self.sliders[prop].setValue(int(value * 10))
                    self.update_property(prop, value)

            if 'blur' in params:
                self.blur_slider.setValue(params['blur'])
                self.update_blur(params['blur'])

            if 'num_images' in params:
                self.num_images_spinbox.setValue(params['num_images'])

            if 'capture_mode' in params:
                index = self.capture_mode_selector.findText(params['capture_mode'])
                if index != -1:
                    self.capture_mode_selector.setCurrentIndex(index)

            print(f"Parámetros cargados desde {file_path}")

    def start_preview(self):
        self.preview_mode = True
        self.timer.start(100)
        self.toggle_preview_button.setEnabled(False)

    def update_preview(self):
        if self.preview_mode:
            try:
                image = self.camera.TakeSnapshot()
                self.display_image(image)
            except Exception:
                pass

    def display_image(self, image, scale_factor=1):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        bytes_per_line = new_width
        qimage = QImage(resized_image.data, new_width, new_height, bytes_per_line, QImage.Format.Format_Grayscale8)
        self.preview_label.setPixmap(QPixmap.fromImage(qimage))

if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())

#%%

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:33:37 2025

@author: Marina
"""


import sys
import os
import numpy as np
import cv2
import threading
import json
import datetime
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, 
                             QSpinBox, QComboBox, QFileDialog, QGroupBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject

class SimulatedCamera:
    def TakeSnapshot(self):
        # Simula una imagen: ruido gaussiano + patrón
        image = np.random.normal(127, 30, (480, 640)).astype(np.uint8)
        cv2.putText(image, 'SIMULATED', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,), 3, cv2.LINE_AA)
        return image

    def set_properties(self, **kwargs):
        pass


# Try to import Lucam
try:
    from lucam import Lucam
    LUCAM_AVAILABLE = True
except ImportError:
    LUCAM_AVAILABLE = False
    print("[WARNING] Módulo 'lucam' no disponible, usando modo simulación.")

class Worker(QObject):
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
                images.append(image)

            if not images:
                print("[ERROR] No se capturó ninguna imagen válida.")
                return

            # Convertimos a float para el promedio si es necesario
            if self.mode == "Promedio":
                stack = np.stack(images).astype(np.float32)
                result_image = np.mean(stack, axis=0).astype(np.uint8)
            elif self.mode == "Mediana":
                stack = np.stack(images).astype(np.uint8)
                result_image = np.median(stack, axis=0).astype(np.uint8)
            else:
                print(f"[WARNING] Modo desconocido: {self.mode}, se usa la primera imagen.")
                result_image = images[0]

            if self.blur_strength > 0:
                k = self.blur_strength * 2 + 1
                result_image = cv2.GaussianBlur(result_image, (k, k), 0)
                
            self.image_captured.emit(result_image)

        except Exception as e:
            pass

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        # Inicialización segura: si falla Lucam, pasamos a SimulatedCamera
        try:
            self.camera = Lucam()
            self.simulation = False
            print("[INFO] Cámara Lucam inicializada correctamente.")
        except Exception as e:
            print(f"[WARNING] No se pudo inicializar Lucam. Se usará SimulatedCamera. Error: {e}")
            self.camera = SimulatedCamera()
            self.simulation = True

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.preview_mode = True
        self.timer.start(300)
        self.capture_mode = "Promedio"
        self.captured_image = None
        self.blur_strength = 0
        self.background_image = None
        self.background_enabled = True

        self.properties = {
            "brightness": (0, 100, 10.0),
            "contrast": (0, 10, 1.0),
            "saturation": (0, 100, 10.0),
            "hue": (-180, 180, 0.0),
            "gamma": (1, 50, 10),
            "exposure": (1, 100, 10.0),
            "gain": (0, 10, 1.0)
        }

        self.work_dir = ""
        self.auto_save = False

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Lumenera Camera Control")
        self.setGeometry(100, 100, 1000, 600)

        self.preview_label = QLabel("Preview en vivo")
        self.preview_label.setFixedSize(640, 480)

        controls_layout = QVBoxLayout()

        #Sección Directorio de Trabajo
        dir_layout = QHBoxLayout()
        self.dir_line_edit = QLineEdit()
        self.dir_button = QPushButton("...")
        self.dir_button.setFixedWidth(30)
        self.dir_button.clicked.connect(self.select_work_dir)
        dir_layout.addWidget(QLabel("Directorio:"))
        dir_layout.addWidget(self.dir_line_edit)
        dir_layout.addWidget(self.dir_button)

        #Sección Guardado Automático
        auto_save_layout = QHBoxLayout()
        self.auto_save_selector = QComboBox()
        self.auto_save_selector.addItems(["No", "Sí"])
        self.auto_save_selector.currentTextChanged.connect(self.toggle_auto_save)
        auto_save_layout.addWidget(QLabel("Guardar automáticamente:"))
        auto_save_layout.addWidget(self.auto_save_selector)

        #Botones principales
        self.capture_button = QPushButton("Capturar Imagen")
        self.capture_button.clicked.connect(self.capture_image)

        self.capture_background_button = QPushButton("Capturar Fondo")
        self.capture_background_button.clicked.connect(self.capture_background)

        #Nueva sección: Restar fondo
        toggle_background_layout = QHBoxLayout()
        toggle_background_label = QLabel("Restar fondo:")
        self.toggle_background_selector = QComboBox()
        self.toggle_background_selector.addItems(["Sí", "No"])
        self.toggle_background_selector.currentTextChanged.connect(self.toggle_background)
        toggle_background_layout.addWidget(toggle_background_label)
        toggle_background_layout.addWidget(self.toggle_background_selector)

        self.save_button = QPushButton("Guardar Imagen")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)

        self.toggle_preview_button = QPushButton("Volver a Preview")
        self.toggle_preview_button.clicked.connect(self.start_preview)
        self.toggle_preview_button.setEnabled(False)

        self.save_settings_button = QPushButton("Guardar Parámetros")
        self.save_settings_button.clicked.connect(self.save_parameters)

        self.load_settings_button = QPushButton("Cargar Parámetros")
        self.load_settings_button.clicked.connect(self.load_parameters)

        #Ajustes de Captura
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

        #Blur
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

        #Controles de Cámara
        camera_controls_box = QGroupBox("Controles de la Cámara")
        camera_controls_layout = QVBoxLayout()

        self.sliders = {}
        self.inputs = {}
        for prop, (min_val, max_val, default) in self.properties.items():
            label = QLabel(f"{prop.capitalize()}: {default}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(min_val * 10))
            slider.setMaximum(int(max_val * 10))
            slider.setValue(int(default * 10))
            slider.valueChanged.connect(lambda value, p=prop: self.update_property(p, value / 10))

            input_field = QLineEdit(str(default))
            input_field.setFixedWidth(50)
            input_field.editingFinished.connect(lambda p=prop, field=input_field: self.set_property_from_input(p, field))

            hbox = QHBoxLayout()
            hbox.addWidget(label)
            hbox.addWidget(slider)
            hbox.addWidget(input_field)

            camera_controls_layout.addLayout(hbox)
            self.sliders[prop] = slider
            self.inputs[prop] = input_field

        camera_controls_box.setLayout(camera_controls_layout)

        #Armado de Layout Principal
        controls_layout.addLayout(dir_layout)
        controls_layout.addLayout(auto_save_layout)
        controls_layout.addWidget(capture_settings_box)
        controls_layout.addWidget(self.capture_button)
        controls_layout.addWidget(self.capture_background_button)
        controls_layout.addLayout(toggle_background_layout)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.toggle_preview_button)
        controls_layout.addWidget(self.save_settings_button)
        controls_layout.addWidget(self.load_settings_button)
        controls_layout.addWidget(camera_controls_box)
        controls_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.preview_label)
        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)


    def update_property(self, prop, value):
        if self.camera:
            self.camera.set_properties(**{prop: value})
        self.inputs[prop].setText(f"{value:.1f}")

    def set_property_from_input(self, prop, field):
        try:
            value = float(field.text())
            self.sliders[prop].setValue(int(value * 10))
            self.update_property(prop, value)
        except ValueError:
            field.setText(f"{self.sliders[prop].value() / 10:.1f}")

    def update_blur(self, value):
        self.blur_strength = value
        self.blur_label.setText(f"Blur: {value}")
        self.blur_input.setText(str(value))

    def set_blur_from_input(self):
        try:
            value = int(self.blur_input.text())
            if 0 <= value <= 10:
                self.blur_slider.setValue(value)
            else:
                self.blur_input.setText(str(self.blur_slider.value()))
        except ValueError:
            self.blur_input.setText(str(self.blur_slider.value()))

    def change_capture_mode(self, mode):
        self.capture_mode = mode

    def select_work_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio de Trabajo")
        if directory:
            self.work_dir = directory
            self.dir_line_edit.setText(directory)
            print(f"[INFO] Directorio de trabajo establecido: {directory}")

    def toggle_auto_save(self, text):
        self.auto_save = (text == "Sí")
        print(f"[INFO] Guardado automático {'activado' if self.auto_save else 'desactivado'}.")

    def save_image_automatically(self, image, tipo):
        if not self.work_dir:
            print("[ERROR] No se definió directorio de trabajo.")
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tipo}_{timestamp}.tif"
        full_path = os.path.join(self.work_dir, filename)
        success = cv2.imwrite(full_path, image)
        if success:
            print(f"[INFO] Imagen guardada automáticamente: {full_path}")
        else:
            print(f"[ERROR] No se pudo guardar la imagen en {full_path}")

    def capture_background(self):
        self.preview_mode = False
        self.timer.stop()
        num_images = self.num_images_spinbox.value()
        mode = self.capture_mode_selector.currentText()
        blur_strength = self.blur_strength
        self.background_worker = Worker(self.camera, num_images, mode, blur_strength)
        self.background_worker.image_captured.connect(self.set_background_image)
        threading.Thread(target=self.background_worker.run, daemon=True).start()

    def set_background_image(self, image):
        self.background_image = image
        self.display_image(image)
        print("Fondo capturado.")
        if self.auto_save:
            self.save_image_automatically(image, "fondo")

    def toggle_background(self, text):
        self.background_enabled = (text == "No")
       
    def capture_image(self):
        num_images = self.num_images_spinbox.value()
        mode = self.capture_mode_selector.currentText()
        if self.simulation:
            print("[INFO] Captura de imagen simulada.")
        self.preview_mode = False
        self.timer.stop()
        self.worker = Worker(self.camera, num_images, mode, self.blur_strength)
        self.worker.image_captured.connect(self.display_captured_image)
        threading.Thread(target=self.worker.run, daemon=True).start()

    def display_captured_image(self, image):
        if (self.background_enabled 
                and self.background_image is not None 
                and self.background_image.shape == image.shape):
            image_int16 = image.astype(np.int16)
            background_int16 = self.background_image.astype(np.int16)
            diff = image_int16 - background_int16
            diff_shifted = diff + 255
            diff_normalized = np.clip((diff_shifted / 2), 0, 255).astype(np.uint8)
            result = diff_normalized
            tipo_guardado = "resta"
        else:
            result = image.copy()
            tipo_guardado = "normal"

        self.captured_image = result
        self.display_image(result)
        self.save_button.setEnabled(True)
        self.toggle_preview_button.setEnabled(True)

        if self.auto_save:
            self.save_image_automatically(result, tipo_guardado)

    def save_image(self):
        if self.captured_image is None:
            return
        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix("tif")
        file_path, _ = file_dialog.getSaveFileName(self, "Guardar Imagen", "", "TIFF (*.tif)")
        if file_path:
            cv2.imwrite(file_path, self.captured_image)
            print(f"Imagen guardada en {file_path}")

    def save_parameters(self):
        params = {prop: self.sliders[prop].value() / 10 for prop in self.properties}
        params['blur'] = self.blur_slider.value()
        params['num_images'] = self.num_images_spinbox.value()
        params['capture_mode'] = self.capture_mode_selector.currentText()
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Parámetros", "", "JSON (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"Parámetros guardados en {file_path}")

    def load_parameters(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Cargar Parámetros", "", "JSON (*.json)")
        if file_path:
            with open(file_path, 'r') as f:
                params = json.load(f)
            for prop in self.properties:
                value = params.get(prop, None)
                if value is not None:
                    self.sliders[prop].setValue(int(value * 10))
                    self.update_property(prop, value)
            if 'blur' in params:
                self.blur_slider.setValue(params['blur'])
                self.update_blur(params['blur'])
            if 'num_images' in params:
                self.num_images_spinbox.setValue(params['num_images'])
            if 'capture_mode' in params:
                index = self.capture_mode_selector.findText(params['capture_mode'])
                if index != -1:
                    self.capture_mode_selector.setCurrentIndex(index)
            print(f"Parámetros cargados desde {file_path}")

    def start_preview(self):
        self.preview_mode = True
        self.timer.start(100)
        self.toggle_preview_button.setEnabled(False)

    def update_preview(self):
        if self.preview_mode:
            try:
                image = self.camera.TakeSnapshot()
                if image is not None:
                    self.display_image(image)
                else:
                    print("[WARNING] TakeSnapshot devolvió None")
                    
                if self.simulation:
                    self.setWindowTitle("Lumenera Camera Control [Simulación]")
                else:
                    self.setWindowTitle("Lumenera Camera Control")
            except Exception:
                pass

    def display_image(self, image, scale_factor=1):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        bytes_per_line = new_width
        qimage = QImage(resized_image.data, new_width, new_height, bytes_per_line, QImage.Format.Format_Grayscale8)
        self.preview_label.setPixmap(QPixmap.fromImage(qimage))

#Main
if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())