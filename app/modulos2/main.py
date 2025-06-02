import sys
from camera_app import CameraApp
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())