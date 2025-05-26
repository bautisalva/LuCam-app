import os
import datetime
import threading
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QSpinBox, QDoubleSpinBox, QLineEdit, QCheckBox, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from image_utils import to_8bit_for_preview
from roi_utils import aplicar_roi, validar_roi
from logging_utils import Logger


class CaptureTab(QWidget):
    def __init__(self, capture_manager):
        super().__init__()
        self.manager = capture_manager

        self.preview_label = QLabel("Imagen capturada")
        self.preview_label.setFixedSize(960, 720)

        self.num_images_input = QSpinBox()
        self.num_images_input.setRange(1, 50)
        self.num_images_input.setValue(5)

        self.blur_input = QSpinBox()
        self.blur_input.setRange(0, 20)
        self.blur_input.setValue(0)

        self.capture_mode_input = QLineEdit("Promedio")

        self.auto_save_checkbox = QCheckBox("Guardar automáticamente")

        self.roi_checkbox = QCheckBox("Aplicar ROI")
        self.roi_x_input = QSpinBox(); self.roi_x_input.setMaximum(9999)
        self.roi_y_input = QSpinBox(); self.roi_y_input.setMaximum(9999)
        self.roi_width_input = QSpinBox(); self.roi_width_input.setMaximum(9999)
        self.roi_height_input = QSpinBox(); self.roi_height_input.setMaximum(9999)

        self.bg_gain_input = QDoubleSpinBox(); self.bg_gain_input.setValue(1.0)
        self.bg_offset_input = QDoubleSpinBox(); self.bg_offset_input.setValue(0.0)

        self.work_dir_input = QLineEdit()
        self.select_dir_button = QPushButton("Seleccionar carpeta")
        self.select_dir_button.clicked.connect(self.select_folder)

        self.capture_button = QPushButton("Capturar Imagen")
        self.capture_button.clicked.connect(self.capture_image)

        self.capture_bg_button = QPushButton("Capturar Fondo")
        self.capture_bg_button.clicked.connect(self.capture_background)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(self.preview_label)

        param_layout = QHBoxLayout()

        roi_group = QGroupBox("ROI")
        roi_layout = QVBoxLayout()
        roi_layout.addWidget(self.roi_checkbox)
        roi_layout.addWidget(QLabel("X:")); roi_layout.addWidget(self.roi_x_input)
        roi_layout.addWidget(QLabel("Y:")); roi_layout.addWidget(self.roi_y_input)
        roi_layout.addWidget(QLabel("W:")); roi_layout.addWidget(self.roi_width_input)
        roi_layout.addWidget(QLabel("H:")); roi_layout.addWidget(self.roi_height_input)
        roi_group.setLayout(roi_layout)

        bg_group = QGroupBox("Fondo")
        bg_layout = QVBoxLayout()
        bg_layout.addWidget(QLabel("Ganancia:")); bg_layout.addWidget(self.bg_gain_input)
        bg_layout.addWidget(QLabel("Offset:")); bg_layout.addWidget(self.bg_offset_input)
        bg_group.setLayout(bg_layout)

        capture_group = QGroupBox("Captura")
        capture_layout = QVBoxLayout()
        capture_layout.addWidget(QLabel("# Imágenes:")); capture_layout.addWidget(self.num_images_input)
        capture_layout.addWidget(QLabel("Blur:")); capture_layout.addWidget(self.blur_input)
        capture_layout.addWidget(QLabel("Modo:")); capture_layout.addWidget(self.capture_mode_input)
        capture_layout.addWidget(self.auto_save_checkbox)
        capture_group.setLayout(capture_layout)

        param_layout.addWidget(roi_group)
        param_layout.addWidget(bg_group)
        param_layout.addWidget(capture_group)

        layout.addLayout(param_layout)

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Directorio de trabajo:"))
        dir_layout.addWidget(self.work_dir_input)
        dir_layout.addWidget(self.select_dir_button)
        layout.addLayout(dir_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.capture_bg_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de trabajo")
        if folder:
            self.work_dir_input.setText(folder)

    def get_roi_config(self):
        return {
            'x': self.roi_x_input.value(),
            'y': self.roi_y_input.value(),
            'w': self.roi_width_input.value(),
            'h': self.roi_height_input.value(),
            'enabled': self.roi_checkbox.isChecked()
        }

    def display_image(self, image):
        image_8bit = to_8bit_for_preview(image)
        height, width = image_8bit.shape
        qimage = QImage(image_8bit.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.preview_label.setPixmap(pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio))

    def capture_image(self):
        self.manager.capture_mode = self.capture_mode_input.text()
        self.manager.blur_strength = self.blur_input.value()
        self.manager.auto_save = self.auto_save_checkbox.isChecked()
        self.manager.work_dir = self.work_dir_input.text()
        roi_cfg = self.get_roi_config()
        self.manager.capture_image(
            num_images=self.num_images_input.value(),
            blur_strength=self.blur_input.value(),
            roi_config=roi_cfg,
            callback=self.display_image
        )

    def capture_background(self):
        self.manager.blur_strength = self.blur_input.value()
        self.manager.capture_mode = self.capture_mode_input.text()
        self.manager.capture_background(
            num_images=self.num_images_input.value(),
            blur_strength=self.blur_input.value(),
            callback=self.display_image
        )
