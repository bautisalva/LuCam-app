import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget
from PyQt5.QtCore import QThread

from gui_preview import PreviewPanel
from gui_analysis import AnalysisTab
from gui_capture import CaptureManager, CaptureTab
from camera_interface import get_camera, API
from logging_utils import Logger

from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from image_utils import to_8bit_for_preview
from PyQt5.QtGui import QImage, QPixmap
import numpy as np


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        self.logger = Logger()
        self.camera, self.simulation = get_camera()

        self.frameformat, fps = self.camera.GetFormat()
        self.frameformat.pixelFormat = API.LUCAM_PF_16
        self.camera.SetFormat(self.frameformat, fps)
        self.camera.ContinuousAutoExposureDisable()

        self.frame_width = self.frameformat.width
        self.frame_height = self.frameformat.height

        try:
            self.available_fps = self.camera.EnumAvailableFrameRates()
        except Exception:
            self.available_fps = [7.5, 15.0, 30.0]

        self.properties = {
            "brightness": (0, 100, 10.0),
            "contrast": (0, 10, 1.0),
            "saturation": (0, 100, 10.0),
            "hue": (-180, 180, 0.0),
            "gamma": (1, 5, 10),
            "exposure": (1, 375, 10.0),
            "gain": (0, 7.75, 1.0),
        }

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Lumenera Camera Control")
        self.setGeometry(100, 100, 1000, 600)

        self.tabs = QTabWidget(self)

        self.preview_tab = PreviewPanel(
            camera=self.camera,
            properties=self.properties,
            available_fps=self.available_fps,
            apply_property_cb=self.apply_property,
            load_cb=self.load_parameters,
            save_cb=self.save_parameters,
            refresh_cb=self.refresh_properties_from_camera,
            logger=self.logger
        )

        self.capture_manager = CaptureManager(camera=self.camera, logger=self.logger)
        self.capture_tab = CaptureTab(capture_manager=self.capture_manager)

        self.analysis_tab = AnalysisTab(get_image_callback=self.get_last_image)

        self.tabs.addTab(self.preview_tab, "Preview")
        self.tabs.addTab(self.capture_tab, "Captura")
        self.tabs.addTab(self.analysis_tab, "Análisis")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def apply_property(self, prop, value):
        if self.camera:
            self.camera.set_properties(**{prop: value})
        self.logger.log(f"Propiedad '{prop}' actualizada a {value:.2f}")

    def refresh_properties_from_camera(self):
        for prop in self.properties:
            try:
                value, _ = self.camera.GetProperty(prop)
                self.preview_tab.sliders[prop].setValue(int(value * 100))
                self.preview_tab.inputs[prop].setText(f"{value:.2f}")
            except Exception as e:
                self.logger.log(f"[ERROR] No se pudo leer propiedad '{prop}': {e}")

    def save_parameters(self):
        pass  # Delegar en config_io

    def load_parameters(self):
        pass  # Delegar en config_io

    def get_last_image(self):
        return getattr(self.capture_manager, "last_full_image", None)

    def closeEvent(self, event):
        self.logger.close()
        event.accept()
