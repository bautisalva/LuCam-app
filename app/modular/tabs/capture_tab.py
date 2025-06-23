from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QSlider, QLabel, QLineEdit, QComboBox, QPushButton, QPlainTextEdit,
    QSpinBox, QFileDialog
)
from PyQt5.QtCore import Qt
import datetime
import os
import json
import numpy as np
from skimage.io import imsave
from common import ROILabel
from utils import blur_uint16, to_8bit_for_preview
from PyQt5.QtGui import QImage, QPixmap
from scipy.ndimage import gaussian_filter
import threading
from skimage.transform import resize
from skimage.color import rgb2gray


class CaptureTab(QWidget):
    def __init__(self, camera, log_message, get_last_image, simulation=False, frame_width=640, frame_height=480):
        super().__init__()
        self.camera = camera
        self.log_message = log_message
        self.get_last_image = get_last_image
        self.simulation = simulation
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.init_ui()
        self.setup_state()

    def setup_state(self):
        self.roi_enabled = False
        self.last_full_image = None
        self.captured_image = None
        self.background_image = None
        self.background_enabled = True
        self.background_gain = 1.0
        self.background_offset = 0.0
        self.blur_strength = 0
        self.capture_mode = "Promedio"
        self.auto_save = False
        self.work_dir = ""
        self.preview_mode = True

    def init_ui(self):
        layout = QHBoxLayout()

        # Lado izquierdo: preview y consola
        left_layout = QVBoxLayout()
        self.preview_label = ROILabel()
        self.preview_label.setFixedSize(960, 720)
        self.preview_label.roi_selected.connect(self.set_roi_from_mouse)
        left_layout.addWidget(self.preview_label)
        
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        left_layout.addWidget(self.console)
        layout.addLayout(left_layout)

        # Controles
        controls_layout = QVBoxLayout()
        
        # Directorio de trabajo
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_button = QPushButton("...")
        self.dir_button.setFixedWidth(30)
        self.dir_button.clicked.connect(self.select_work_dir)
        dir_layout.addWidget(QLabel("Directorio:"))
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(self.dir_button)
        controls_layout.addLayout(dir_layout)
        
        # Auto-guardado
        auto_save_layout = QHBoxLayout()
        self.auto_save_combo = QComboBox()
        self.auto_save_combo.addItems(["No", "Sí"])
        self.auto_save_combo.currentTextChanged.connect(self.toggle_auto_save)
        auto_save_layout.addWidget(QLabel("Guardar automáticamente:"))
        auto_save_layout.addWidget(self.auto_save_combo)
        controls_layout.addLayout(auto_save_layout)

        # Configuraciones de captura
        capture_settings_box = QGroupBox("Configuraciones de Captura")
        capture_settings_layout = QVBoxLayout()
        
        # Número de imágenes
        capture_num_layout = QHBoxLayout()
        self.num_images_spinbox = QSpinBox()
        self.num_images_spinbox.setRange(1, 100)
        self.num_images_spinbox.setValue(5)
        capture_num_layout.addWidget(QLabel("Imágenes por captura:"))
        capture_num_layout.addWidget(self.num_images_spinbox)
        capture_settings_layout.addLayout(capture_num_layout)
        
        # Modo de captura
        capture_mode_layout = QHBoxLayout()
        self.capture_mode_selector = QComboBox()
        self.capture_mode_selector.addItems(["Promedio", "Mediana"])
        self.capture_mode_selector.currentTextChanged.connect(self.change_capture_mode)
        capture_mode_layout.addWidget(QLabel("Modo de captura:"))
        capture_mode_layout.addWidget(self.capture_mode_selector)
        capture_settings_layout.addLayout(capture_mode_layout)
        
        # Blur
        blur_box = QGroupBox("Desenfoque")
        blur_layout = QHBoxLayout()
        self.blur_label = QLabel("Blur: 0")
        self.blur_slider = QSlider(Qt.Horizontal)
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
        capture_settings_layout.addWidget(blur_box)
        
        capture_settings_box.setLayout(capture_settings_layout)
        controls_layout.addWidget(capture_settings_box)
        
        # Botones de captura
        self.capture_button = QPushButton("Capturar Imagen")
        self.capture_button.clicked.connect(self.capture_image)
        controls_layout.addWidget(self.capture_button)
        
        self.capture_background_button = QPushButton("Capturar Fondo")
        self.capture_background_button.clicked.connect(self.capture_background)
        controls_layout.addWidget(self.capture_background_button)
        
        # Toggle fondo
        toggle_background_layout = QHBoxLayout()
        toggle_background_label = QLabel("Restar fondo:")
        self.toggle_background_selector = QComboBox()
        self.toggle_background_selector.addItems(["Sí", "No"])
        self.toggle_background_selector.currentTextChanged.connect(self.toggle_background)
        toggle_background_layout.addWidget(toggle_background_label)
        toggle_background_layout.addWidget(self.toggle_background_selector)
        controls_layout.addLayout(toggle_background_layout)
        
        # Ganancia y offset
        gain_offset_box = QGroupBox("Fondo: Ganancia y Offset")
        gain_offset_layout = QVBoxLayout()
        
        # Ganancia
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Ganancia (a):"))
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setMinimum(0)
        self.gain_slider.setMaximum(10000)
        self.gain_slider.setValue(100)
        self.gain_slider.valueChanged.connect(lambda v: self.update_gain(v / 100))
        gain_layout.addWidget(self.gain_slider)
        self.gain_input = QLineEdit("1.0")
        self.gain_input.setFixedWidth(50)
        self.gain_input.editingFinished.connect(self.set_gain_from_input)
        gain_layout.addWidget(self.gain_input)
        
        # Offset
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Offset (b):"))
        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setMinimum(-100)
        self.offset_slider.setMaximum(100)
        self.offset_slider.setValue(0)
        self.offset_slider.valueChanged.connect(lambda v: self.update_offset(v / 100))
        offset_layout.addWidget(self.offset_slider)
        self.offset_input = QLineEdit("0")
        self.offset_input.setFixedWidth(50)
        self.offset_input.editingFinished.connect(self.set_offset_from_input)
        offset_layout.addWidget(self.offset_input)
        
        gain_offset_layout.addLayout(gain_layout)
        gain_offset_layout.addLayout(offset_layout)
        gain_offset_box.setLayout(gain_offset_layout)
        controls_layout.addWidget(gain_offset_box)
        
        # Botones adicionales
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
        self.setLayout(layout)

    def toggle_roi(self, text):
        self.roi_enabled = (text == "Sí")
        estado = "activado" if self.roi_enabled else "desactivado"
        self.log_message(f"Uso de ROI {estado}")
        
        if self.last_full_image is not None:
            self.display_captured_image_in_tab(self.last_full_image)

    def set_roi_from_mouse(self, x, y, w, h):
        # Get current camera format
        frameformat, fps = self.camera.GetFormat()
        max_width = frameformat.width
        max_height = frameformat.height
        
        # Calculate scaling factor
        scale_x = max_width / self.preview_label.width()
        scale_y = max_height / self.preview_label.height()
        
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
        
        self.log_message(f"ROI seleccionado: x={x}, y={y}, w={w}, h={h}")

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
        self.background_offset = value * 32768  # Mapear de [-1,1] a [-32768,32768]
        self.offset_input.setText(f"{value:.2f}")
        self.log_message(f"Offset del fondo ajustado a {self.background_offset:.0f} (escalado: {value:.2f})")

    def set_offset_from_input(self):
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
        self.capture_mode = mode
        self.log_message(f"Modo de captura cambiado a {mode}")

    def select_work_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio de Trabajo")
        if directory:
            self.work_dir = directory
            self.dir_edit.setText(directory)
            self.log_message(f"Directorio de trabajo establecido: {directory}")

    def toggle_auto_save(self, text):
        self.auto_save = (text == "Sí")
        estado = "activado" if self.auto_save else "desactivado"
        self.log_message(f"Guardado automático {estado}")

    def capture_background(self):
        num_images = self.num_images_spinbox.value()
        mode = self.capture_mode_selector.currentText()
        blur_strength = self.blur_strength
        
        # Worker para captura en segundo plano
        from common import Worker
        self.background_worker = Worker(self.camera, num_images, mode, blur_strength)
        self.background_worker.image_captured.connect(self.set_background_image)
        threading.Thread(target=self.background_worker.run, daemon=True).start()
        self.log_message("Iniciando captura de fondo...")

    def set_background_image(self, image):
        self.background_image = image
        self.last_full_image = image.copy()
        self.log_message("Fondo capturado correctamente.")
        
        # Aplicar ROI si está habilitado
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
            self.log_message(f"[INFO] ROI aplicado al fondo: x={x}, y={y}, w={w}, h={h}")
        else:
            image_roi = image.copy()
            if self.roi_enabled:
                self.log_message("[WARNING] ROI inválido al capturar fondo. Mostrando imagen completa.")
        
        self.display_captured_image_in_tab(image_roi)
        
        if self.auto_save and self.work_dir:
            fondo_path = os.path.join(self.work_dir, f"fondo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tif")
            imsave(fondo_path, image)
            self.log_message(f"Imagen de fondo guardada en: {fondo_path}")

    def toggle_background(self, text):
        self.background_enabled = (text == "Sí")
        estado = "activada" if self.background_enabled else "desactivada"
        self.log_message(f"Restar fondo {estado}")

    def apply_background_subtraction(self, image_roi, x=0, y=0, w=None, h=None):
        if (self.background_image is None or
            w is None or h is None or
            self.background_image.shape[0] < y + h or
            self.background_image.shape[1] < x + w):
            return image_roi
        
        bg_roi = self.background_image[y:y+h, x:x+w]
        a = self.background_gain
        b = self.background_offset
        
        image_float = image_roi.astype(np.float32)
        bg_float = bg_roi.astype(np.float32)
        
        diff = a * (image_float - bg_float + b)
        diff_centered = diff + 32768
        return np.clip(diff_centered, 0, 65535).astype(np.uint16)

    def capture_image(self):
        num_images = self.num_images_spinbox.value()
        mode = self.capture_mode_selector.currentText()
        
        # Worker para captura en segundo plano
        from common import Worker
        self.worker = Worker(self.camera, num_images, mode, self.blur_strength)
        self.worker.image_captured.connect(self.display_captured_image)
        threading.Thread(target=self.worker.run, daemon=True).start()
        self.log_message("Iniciando captura de imagen...")

    def display_captured_image(self, image):
        self.last_full_image = image.copy()
        
        # Aplicar ROI si está habilitado
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
        
        self.captured_image = image_roi
        
        # Aplicar resta de fondo si está habilitado
        if self.background_enabled and self.background_image is not None:
            display_image = self.apply_background_subtraction(image_roi, x, y, w, h)
        else:
            display_image = image_roi
        
        self.display_captured_image_in_tab(display_image)
        self.save_button.setEnabled(True)
        self.toggle_preview_button.setEnabled(True)
        self.log_message("Imagen capturada y mostrada en pestaña 'Captura'.")
        
        if self.auto_save:
            self.save_auto_images(
                full_image=image.copy(),
                roi_image=image_roi.copy(),
                roi_coords=(x, y, w, h)
            )

    def save_auto_images(self, full_image, roi_image, roi_coords):
        x, y, w, h = roi_coords
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Guardar imagen cruda (completa)
        raw_folder = os.path.join(self.work_dir, "raw")
        os.makedirs(raw_folder, exist_ok=True)
        raw_path = os.path.join(raw_folder, f"cruda_{timestamp}.tif")
        imsave(raw_path, full_image)
        self.log_message(f"Imagen cruda (completa) guardada en: {raw_path}")
        
        # 2. Guardar imagen normal (ROI + blur)
        normal_image = roi_image
        if self.blur_strength > 0:
            normal_image = blur_uint16(normal_image, sigma=self.blur_strength)
        
        normal_path = os.path.join(self.work_dir, f"normal_{timestamp}.tif")
        imsave(normal_path, normal_image)
        self.log_message(f"Imagen normal (ROI + blur) guardada en: {normal_path}")
        
        # 3. Guardar imagen resta (ROI + blur + fondo)
        if self.background_enabled and self.background_image is not None:
            resta_image = self.apply_background_subtraction(
                normal_image, x, y, w, h
            )
            resta_path = os.path.join(self.work_dir, f"resta_{timestamp}.tif")
            imsave(resta_path, resta_image)
            self.log_message(f"Imagen resta (ROI + blur + fondo) guardada en: {resta_path}")
        else:
            self.log_message("Fondo no activado o no disponible, imagen 'resta' no guardada.")

    def display_captured_image_in_tab(self, image):
        if image is None:
            return
            
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            image = (rgb2gray(image) * 65535).astype(np.uint16)
        
        # Redimensionar para mostrar
        height, width = image.shape
        target_width = self.preview_label.width()
        target_height = self.preview_label.height()
        
        # Mantener relación de aspecto
        if width / height > target_width / target_height:
            new_width = target_width
            new_height = int(target_width * height / width)
        else:
            new_height = target_height
            new_width = int(target_height * width / height)
        
        # Convertir a 8-bit para visualización
        resized_image = resize(image, (new_height, new_width), preserve_range=True).astype(np.uint16)
        image_8bit = to_8bit_for_preview(resized_image)
        
        # Crear QImage y QPixmap
        qimage = QImage(
            image_8bit.data, 
            image_8bit.shape[1], 
            image_8bit.shape[0], 
            image_8bit.shape[1], 
            QImage.Format_Grayscale8
        )
        pixmap = QPixmap.fromImage(qimage)
        
        # Centrar la imagen en el label
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setPixmap(pixmap)

    def save_image(self):
        if self.captured_image is None:
            self.log_message("[ERROR] No hay imagen capturada para guardar.")
            return
            
        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix("tif")
        file_path, _ = file_dialog.getSaveFileName(
            self, "Guardar Imagen", "", "TIFF (*.tif)"
        )
        
        if file_path:
            imsave(file_path, self.captured_image)
            self.log_message(f"Imagen guardada manualmente en: {file_path}")

    def save_capture_parameters(self):
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
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Parámetros de Captura", "", "JSON (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(params, f, indent=4)
                self.log_message(f"Parámetros de Captura guardados en {file_path}")
            except Exception as e:
                self.log_message(f"[ERROR] No se pudieron guardar parámetros de Captura: {e}")

    def load_parameters(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Parámetros", "", "JSON (*.json)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                params = json.load(f)
                
            # Aplicar parámetros
            if 'blur_strength' in params:
                self.blur_slider.setValue(params['blur_strength'])
                self.update_blur(params['blur_strength'])
                
            if 'num_images' in params:
                self.num_images_spinbox.setValue(params['num_images'])
                
            if 'background_gain' in params:
                self.gain_slider.setValue(int(params['background_gain'] * 100))
                self.update_gain(params['background_gain'])
                
            if 'background_offset' in params:
                self.offset_slider.setValue(int(params['background_offset'] * 100))
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
                
            self.log_message(f"Parámetros cargados desde: {file_path}")
            
        except Exception as e:
            self.log_message(f"[ERROR] Error cargando parámetros: {e}")

    def start_preview(self):
        self.toggle_preview_button.setEnabled(False)
        self.log_message("Preview reanudado.")