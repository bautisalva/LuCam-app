import sys
import os
import numpy as np
import cv2
import threading
import json
import datetime
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, 
                             QSpinBox, QComboBox, QFileDialog,
                             QGroupBox, QTabWidget, QGridLayout,QPlainTextEdit)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread

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

        except Exception:
            pass
        
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
            QThread.msleep(300)  # delay entre frames

    def stop(self):
        self.running = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


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
            
        self.log_file_path = os.path.join(os.getcwd(), "log.txt")
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
        
        # Escribir mensaje de inicio
        now = datetime.datetime.now()
        start_message = f"=== Se inició la app el día {now.strftime('%d/%m/%Y')} a las {now.strftime('%H:%M:%S')} ==="
        self.log_file.write(start_message + "\n")
        self.log_file.flush()  # Asegurarnos que se escriba inmediatamente

        self.preview_worker = PreviewWorker(self.camera)
        self.preview_thread = QThread()
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_worker.new_frame.connect(self.display_preview_image)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_thread.start()

        self.preview_label_preview = QLabel("Preview en vivo")
        self.preview_label_preview.setFixedSize(640, 480)
        
        self.preview_label_capture = QLabel("Preview captura")
        self.preview_label_capture.setFixedSize(640, 480)
        
        self.console_preview = QPlainTextEdit()
        self.console_preview.setReadOnly(True)

        self.console_capture = QPlainTextEdit()
        self.console_capture.setReadOnly(True)

        self.background_gain = 1.0
        self.background_offset = 0.0
        
        self.preview_mode = True
        
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
            "exposure": (1, 500, 10.0),
            "gain": (0, 10, 1.0)
        }

        self.work_dir = ""
        self.auto_save = False

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Lumenera Camera Control")
        self.setGeometry(100, 100, 1000, 600)
    
        # Crear las pestañas
        self.tabs = QTabWidget(self)
    
        # Pestaña 1: Preview
        self.preview_tab = QWidget()
        self.init_preview_tab()
    
        # Pestaña 2: Captura
        self.capture_tab = QWidget()
        self.init_capture_tab()
    
        # Agregar pestañas
        self.tabs.addTab(self.preview_tab, "Preview")
        self.tabs.addTab(self.capture_tab, "Captura")
    
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
        
    def init_preview_tab(self):
        layout = QHBoxLayout()
    
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.preview_label_preview)
        left_layout.addWidget(self.console_preview)
    
        layout.addLayout(left_layout)
        # Controles de la cámara a la derecha
        controls_layout = QVBoxLayout()
        self.sliders = {}
        self.inputs = {}
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
    
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Botones para guardar y cargar parámetros
        self.save_preview_button = QPushButton("Guardar Parámetros de Preview")
        self.save_preview_button.clicked.connect(self.save_preview_parameters)
        controls_layout.addWidget(self.save_preview_button)

        self.load_settings_button = QPushButton("Cargar Parámetros")
        self.load_settings_button.clicked.connect(self.load_parameters)
        
        controls_layout.addWidget(self.load_settings_button)
        self.apply_default_slider_values_to_camera()

        self.preview_tab.setLayout(layout)

    def init_capture_tab(self):
        layout = QHBoxLayout()
    
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.preview_label_capture)
        left_layout.addWidget(self.console_capture)
        
        layout.addLayout(left_layout)
        # Controles a la derecha
        controls_layout = QVBoxLayout()
    
        # Directorio de trabajo
        dir_layout = QHBoxLayout()
        self.dir_line_edit = QLineEdit()
        self.dir_button = QPushButton("...")
        self.dir_button.setFixedWidth(30)
        self.dir_button.clicked.connect(self.select_work_dir)
        dir_layout.addWidget(QLabel("Directorio:"))
        dir_layout.addWidget(self.dir_line_edit)
        dir_layout.addWidget(self.dir_button)
        controls_layout.addLayout(dir_layout)
    
        # Guardado automático
        auto_save_layout = QHBoxLayout()
        self.auto_save_selector = QComboBox()
        self.auto_save_selector.addItems(["No", "Sí"])
        self.auto_save_selector.currentTextChanged.connect(self.toggle_auto_save)
        auto_save_layout.addWidget(QLabel("Guardar automáticamente:"))
        auto_save_layout.addWidget(self.auto_save_selector)
        controls_layout.addLayout(auto_save_layout)
    
        # Ajustes de captura
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
    
        # Blur
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
        # Ganancia y offset para fondo
        gain_offset_box = QGroupBox("Fondo: Ganancia y Offset")
        gain_offset_layout = QVBoxLayout()
        # Botones
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
    
        # Ganancia
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
        self.offset_slider.setMinimum(-255)
        self.offset_slider.setMaximum(255)
        self.offset_slider.setValue(0)
        self.offset_slider.valueChanged.connect(lambda v: self.update_offset(v))
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
            for prop, (min_val, max_val, default) in self.properties.items():
                if self.camera:
                    self.camera.set_properties(**{prop: default})
    
    def update_property(self, prop, value):
        if self.camera:
            self.camera.set_properties(**{prop: value})
        self.sliders[prop].blockSignals(True)
        self.sliders[prop].setValue(int(value * 10))
        self.sliders[prop].blockSignals(False)
        self.inputs[prop].setText(f"{value:.1f}")
        self.log_message(f"Se actualizó '{prop}' a {value:.1f}")

    
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
        self.log_message(f"Se seteó el blur a {value}")
    
    def set_blur_from_input(self):
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
        self.background_gain = value
        self.gain_input.setText(f"{value:.2f}")
        self.log_message(f"Ganancia del fondo ajustada a {value:.2f}")

    def set_gain_from_input(self):
        try:
            value = float(self.gain_input.text())
            self.background_gain = value
            self.gain_slider.setValue(int(value * 100))
        except ValueError:
            self.gain_input.setText(f"{self.background_gain:.2f}")

    def update_offset(self, value):
        self.background_offset = value
        self.offset_input.setText(str(value))
        self.log_message(f"Offset del fondo ajustado a {value}")

    def set_offset_from_input(self):
        try:
            value = int(self.offset_input.text())
            self.background_offset = value
            self.offset_slider.setValue(value)
        except ValueError:
            self.offset_input.setText(str(self.background_offset))
    def change_capture_mode(self, mode):
        self.capture_mode = mode
        self.log_message(f"Modo de captura cambiado a {mode}")
    
    def select_work_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio de Trabajo")
        if directory:
            self.work_dir = directory
            self.dir_line_edit.setText(directory)
            self.log_message(f"Directorio de trabajo establecido: {directory}")
    
    def toggle_auto_save(self, text):
        self.auto_save = (text == "Sí")
        estado = "activado" if self.auto_save else "desactivado"
        self.log_message(f"Guardado automático {estado}")
    
    def save_image_automatically(self, image, tipo):
        if not self.work_dir:
            self.log_message("[ERROR] No se definió directorio de trabajo para guardar imagen.")
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tipo}_{timestamp}.tif"
        full_path = os.path.join(self.work_dir, filename)
        success = cv2.imwrite(full_path, image)
        if success:
            self.log_message(f"Imagen guardada automáticamente en: {full_path}")
        else:
            self.log_message(f"[ERROR] No se pudo guardar la imagen en: {full_path}")
    
    def capture_background(self):
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
        self.background_image = image
        self.display_image(image)
        self.log_message("Fondo capturado correctamente.")
        if self.auto_save:
            self.save_image_automatically(image, "fondo")
    
    def toggle_background(self, text):
        self.background_enabled = (text == "No")
        estado = "desactivada" if self.background_enabled else "activada"
        self.log_message(f"Restar fondo {estado}")
    
    def capture_image(self):
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
        if (self.background_enabled and
                self.background_image is not None and
                self.background_image.shape == image.shape):

            a = self.background_gain
            b = self.background_offset

            image_float = image.astype(np.float32)
            background_float = self.background_image.astype(np.float32)

            diff = a*(image_float - background_float + b)
            diff_centered = diff + 128
            result = np.clip(diff_centered, 0, 255).astype(np.uint8)
            tipo_guardado = "resta"
        else:
            result = image.copy()
            tipo_guardado = "normal"
    
        self.captured_image = result
        self.display_captured_image_in_tab(result)
        self.save_button.setEnabled(True)
        self.toggle_preview_button.setEnabled(True)
    
        self.log_message("Imagen capturada y mostrada en pestaña 'Captura'.")
    
        if self.auto_save:
            self.save_image_automatically(result, tipo_guardado)
    def display_captured_image_in_tab(self, image, scale_factor=1):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        bytes_per_line = new_width
        qimage = QImage(resized_image.data, new_width, new_height, bytes_per_line, QImage.Format.Format_Grayscale8)
    
        # Actualiza sólo el label de la pestaña Captura
        self.preview_label_capture.setPixmap(QPixmap.fromImage(qimage))


    def save_image(self):
        if self.captured_image is None:
            self.log_message("[ERROR] No hay imagen capturada para guardar.")
            return
        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix("tif")
        file_path, _ = file_dialog.getSaveFileName(self, "Guardar Imagen", "", "TIFF (*.tif)")
        if file_path:
            cv2.imwrite(file_path, self.captured_image)
            self.log_message(f"Imagen guardada manualmente en: {file_path}")
    
    def save_preview_parameters(self):
        params = {prop: self.sliders[prop].value() / 10 for prop in self.properties}
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Parámetros de Preview", "", "JSON (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(params, f, indent=4)
                self.log_message(f"Parámetros de Preview guardados en {file_path}")
            except Exception as e:
                self.log_message(f"[ERROR] No se pudieron guardar parámetros de Preview: {e}")


    def save_capture_parameters(self):
        params = {
            "blur": self.blur_slider.value(),
            "num_images": self.num_images_spinbox.value(),
            "capture_mode": self.capture_mode_selector.currentText()
        }
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Parámetros de Captura", "", "JSON (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(params, f, indent=4)
                self.log_message(f"Parámetros de Captura guardados en {file_path}")
            except Exception as e:
                self.log_message(f"[ERROR] No se pudieron guardar parámetros de Captura: {e}")


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
            self.log_message(f"Parámetros cargados desde archivo: {file_path}")
    
    def start_preview(self):
        self.preview_mode = True
        self.toggle_preview_button.setEnabled(False)
        self.preview_worker.resume()
        self.log_message("Preview reanudado.")

    def display_preview_image(self, image, scale_factor=1):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        bytes_per_line = new_width
        qimage = QImage(resized_image.data, new_width, new_height, bytes_per_line, QImage.Format.Format_Grayscale8)
    
        # Sólo actualizar el preview
        self.preview_label_preview.setPixmap(QPixmap.fromImage(qimage))


    def display_image(self, image, scale_factor=1):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        bytes_per_line = new_width
        qimage = QImage(resized_image.data, new_width, new_height, bytes_per_line, QImage.Format.Format_Grayscale8)
    
        # Actualizar ambos previews
        self.preview_label_preview.setPixmap(QPixmap.fromImage(qimage))
        self.preview_label_capture.setPixmap(QPixmap.fromImage(qimage))

    def log_message(self, message):
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
        self.preview_worker.stop()
        self.preview_thread.quit()
        self.preview_thread.wait()
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()
        event.accept()

#Main
if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
    
