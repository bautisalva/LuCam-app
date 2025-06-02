import os
import datetime
import numpy as np
from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout
from tabs.preview_tab import PreviewTab
from tabs.capture_tab import CaptureTab
from tabs.analysis_tab import AnalysisTab
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QRect
from common import SimulatedCamera, PreviewWorker, ROILabel
from utils import log_message as global_log_message

try:
    from lucam import Lucam, API
    LUCAM_AVAILABLE = True
except ImportError:
    LUCAM_AVAILABLE = False
    print("[WARNING] Módulo 'lucam' no disponible, usando modo simulación.")

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_camera()
        self.setup_logging()
        self.setup_preview_worker()
        self.setup_ui()
        
    def setup_camera(self):
        try:
            self.camera = Lucam()
            self.camera.CameraClose()
            self.camera = Lucam()
            self.simulation = False
            print("[INFO] Cámara Lucam reiniciada al iniciar.")
        except Exception as e:
            print(f"[WARNING] Error inicializando Lucam: {e}")
            self.camera = SimulatedCamera()
            self.simulation = True
        
        frameformat, fps = self.camera.GetFormat()
        frameformat.pixelFormat = API.LUCAM_PF_16
        self.camera.SetFormat(frameformat, fps)
        self.frame_width = frameformat.width
        self.frame_height = frameformat.height
        self.camera.ContinuousAutoExposureDisable()
        
        try:
            self.available_fps = self.camera.EnumAvailableFrameRates()
            self.available_fps = [round(f, 2) for f in self.available_fps]
        except:
            self.available_fps = [7.5, 15.0]
            
        self.properties = {
            "brightness": (0, 100, 10.0),
            "contrast": (0, 10, 1.0),
            "saturation": (0, 100, 10.0),
            "hue": (-180, 180, 0.0),
            "gamma": (1, 5, 10),
            "exposure": (1, 375, 10.0),
            "gain": (0, 7.75, 1.0),
        }
        
        self.roi_enabled = False
        self.last_full_image = None

    def setup_logging(self):
        self.log_file_path = os.path.join(os.getcwd(), "log.txt")
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
        now = datetime.datetime.now()
        start_msg = f"=== App iniciada el {now.strftime('%d/%m/%Y')} a las {now.strftime('%H:%M:%S')} ==="
        self.log_file.write(start_msg + "\n")
        self.log_file.flush()

    def setup_preview_worker(self):
        self.preview_worker = PreviewWorker(self.camera)
        self.preview_thread = QThread()
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_worker.new_frame.connect(self.display_preview_image)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_thread.start()

    def setup_ui(self):
        self.setWindowTitle("Lumenera Camera Control")
        self.setGeometry(100, 100, 1000, 600)
        
        self.tabs = QTabWidget(self)
        self.preview_tab = PreviewTab(
            camera=self.camera,
            log_message=self.log_message,
            available_fps=self.available_fps,
            properties=self.properties,
            simulation=self.simulation
        )
        self.capture_tab = CaptureTab(
            camera=self.camera,
            log_message=self.log_message,
            get_last_image=self.get_last_image,
            simulation=self.simulation,
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )
        self.analysis_tab = AnalysisTab(
            get_image_callback=self.get_last_image
        )
        
        self.tabs.addTab(self.preview_tab, "Preview")
        self.tabs.addTab(self.capture_tab, "Captura")
        self.tabs.addTab(self.analysis_tab, "Análisis")
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def get_last_image(self):
        return self.last_full_image


    def log_message(self, message):
        consoles = []
        if hasattr(self, "preview_tab") and hasattr(self.preview_tab, "console"):
            consoles.append(self.preview_tab.console)
        if hasattr(self, "capture_tab") and hasattr(self.capture_tab, "console"):
            consoles.append(self.capture_tab.console)
        if hasattr(self, "analysis_tab") and hasattr(self.analysis_tab, "console"):
            consoles.append(self.analysis_tab.console)
    
        global_log_message(
            message,
            log_file=self.log_file,
            consoles=consoles
        )


    def display_preview_image(self, image):
        self.last_full_image = image.copy()
        self.preview_tab.display_image(image)
        
    def closeEvent(self, event):
        self.preview_worker.stop()
        self.preview_thread.quit()
        self.preview_thread.wait()
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()
        event.accept()