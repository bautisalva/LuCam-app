import sys
from PyQt5.QtWidgets import QApplication
from gui_main import CameraApp

def main():
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
