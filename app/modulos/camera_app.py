from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, 
                             QSpinBox, QComboBox, QFileDialog,
                             QGroupBox, QTabWidget, QGridLayout,QPlainTextEdit)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QRect
from core_camera import init_camera
from widgets_custom import ROILabel
from preview_tab import PreviewTab
from capture_tab import CaptureTab
from analysis_tab import AnalysisTab
from utils import to_8bit_for_preview
from worker_threads import PreviewWorker
import datetime
import os

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        self.camera, self.simulation = init_camera()
        self.available_fps = self.get_available_fps()
        self.properties = {
            "brightness": (0, 100, 10.0),
            "contrast": (0, 10, 1.0),
            "saturation": (0, 100, 10.0),
            "hue": (-180, 180, 0.0),
            "gamma": (1, 5, 10),
            "exposure": (1, 375, 10.0),
            "gain": (0, 7.75, 1.0),
        }

        # Imagen y flags
        self.preview_mode = True
        self.background_enabled = True
        self.blur_strength = 0
        self.capture_mode = "Promedio"
        self.roi_enabled = False
        self.last_full_image = None
        self.background_image = None
        self.background_gain = 1.0
        self.background_offset = 0.0
        self.auto_save = False
        self.work_dir = ""

        # GUI widgets comunes
        self.preview_label_preview = QLabel("Preview en vivo")
        self.preview_label_preview.setFixedSize(960, 720)
        self.console_preview = QPlainTextEdit()
        self.console_preview.setReadOnly(True)

        self.preview_label_capture = ROILabel()
        self.preview_label_capture.setFixedSize(960, 720)
        self.console_capture = QPlainTextEdit()
        self.console_capture.setReadOnly(True)

        # Logging
        self.init_log()

        # Live preview thread
        self.preview_worker = PreviewWorker(self.camera)
        self.preview_thread = QThread()
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_worker.new_frame.connect(self.display_preview_image)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_thread.start()

        # Inicializa tabs
        self.init_ui()

    def init_log(self):
        self.log_file_path = os.path.join(os.getcwd(), "log.txt")
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
        now = datetime.datetime.now()
        msg = f"=== App iniciada el {now.strftime('%d/%m/%Y')} a las {now.strftime('%H:%M:%S')} ==="
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def init_ui(self):
        self.setWindowTitle("LuCam Control")
        self.setGeometry(100, 100, 1000, 600)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(PreviewTab(self), "Preview")
        self.tabs.addTab(CaptureTab(self), "Captura")
        self.tabs.addTab(AnalysisTab(get_image_callback=self.get_last_image), "Análisis")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def log_message(self, text):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        full = f"[{now}] {text}"
        self.console_preview.appendPlainText(full)
        self.console_capture.appendPlainText(full)
        print(full)
        self.log_file.write(full + "\n")
        self.log_file.flush()

    def display_preview_image(self, image):
        from skimage.color import rgb2gray
        from skimage.transform import resize

        if len(image.shape) == 3:
            image = (rgb2gray(image) * 65535).astype(np.uint16)
        image_8bit = to_8bit_for_preview(image)
        from PyQt5.QtGui import QImage, QPixmap
        qimage = QImage(image_8bit.data, image_8bit.shape[1], image_8bit.shape[0], image_8bit.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.preview_label_preview.width(),
            self.preview_label_preview.height(),
            aspectRatioMode=1  # Qt.KeepAspectRatio
        )
        self.preview_label_preview.setPixmap(pixmap)

    def get_last_image(self):
        return self.last_full_image

    def get_available_fps(self):
        try:
            fps_list = self.camera.EnumAvailableFrameRates()
            return [round(f, 2) for f in fps_list]
        except Exception:
            return [7.5, 15.0]

    def closeEvent(self, event):
        self.preview_worker.stop()
        self.preview_thread.quit()
        self.preview_thread.wait()
        if self.log_file:
            self.log_file.close()
        event.accept()

