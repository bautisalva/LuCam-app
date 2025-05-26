from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox,
    QSlider, QLineEdit, QPushButton, QComboBox, QPlainTextEdit
)
from PyQt5.QtCore import Qt
from image_utils import to_8bit_for_preview
from PyQt5.QtGui import QImage, QPixmap

class PreviewPanel(QWidget):
    def __init__(self, camera, properties, available_fps, apply_property_cb, load_cb, save_cb, refresh_cb, logger):
        super().__init__()
        self.camera = camera
        self.properties = properties
        self.available_fps = available_fps
        self.apply_property_cb = apply_property_cb
        self.load_cb = load_cb
        self.save_cb = save_cb
        self.refresh_cb = refresh_cb
        self.logger = logger

        self.sliders = {}
        self.inputs = {}

        self.preview_label = QLabel("Preview en vivo")
        self.preview_label.setFixedSize(960, 720)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)

        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.preview_label)
        left_layout.addWidget(self.console)
        layout.addLayout(left_layout)

        controls_layout = QVBoxLayout()

        fps_group = QGroupBox("FPS")
        fps_layout = QHBoxLayout()
        self.fps_selector = QComboBox()
        self.fps_selector.setStyleSheet("background-color: lightyellow;")
        for fps in self.available_fps:
            self.fps_selector.addItem(f"{fps:.2f}")
        self.fps_selector.currentTextChanged.connect(self.change_fps)
        fps_layout.addWidget(QLabel("Frames por segundo:"))
        fps_layout.addWidget(self.fps_selector)
        fps_group.setLayout(fps_layout)
        controls_layout.addWidget(fps_group)

        for prop, (min_val, max_val, default) in self.properties.items():
            group = QGroupBox(prop.capitalize())
            group_layout = QHBoxLayout()

            label = QLabel(f"{default}")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * 100))
            slider.setMaximum(int(max_val * 100))
            slider.setValue(int(default * 100))
            slider.valueChanged.connect(lambda value, p=prop: self.apply_property_cb(p, value / 100))

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

        self.save_button = QPushButton("Guardar Parámetros de Preview")
        self.save_button.clicked.connect(self.save_cb)
        controls_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Cargar Parámetros")
        self.load_button.clicked.connect(self.load_cb)
        controls_layout.addWidget(self.load_button)

        self.refresh_button = QPushButton("Refrescar desde Cámara")
        self.refresh_button.clicked.connect(self.refresh_cb)
        controls_layout.addWidget(self.refresh_button)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        self.setLayout(layout)

    def set_property_from_input(self, prop, field):
        try:
            value = float(field.text())
            self.sliders[prop].setValue(int(value * 100))
            self.camera.set_properties(**{prop: value})
            self.logger.log(f"{prop} actualizado manualmente a {value:.2f}")
        except ValueError:
            field.setText(f"{self.sliders[prop].value() / 100:.2f}")

    def change_fps(self, fps_text):
        try:
            fps = float(fps_text)
            frameformat, _ = self.camera.GetFormat()
            self.camera.SetFormat(frameformat, fps)
            self.logger.log(f"FPS cambiado a {fps:.2f}")
        except Exception as e:
            self.logger.log(f"[ERROR] No se pudo cambiar el FPS: {e}")

    def update_preview_image(self, image):
        if len(image.shape) == 3:
            image = (rgb2gray(image) * 65535).astype(np.uint16)

        image_8bit = to_8bit_for_preview(image)
        qimage = QImage(image_8bit.data, image_8bit.shape[1], image_8bit.shape[0], image_8bit.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage).scaled(self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio)
        self.preview_label.setPixmap(pixmap)

    def append_log(self, message):
        self.console.appendPlainText(message)
