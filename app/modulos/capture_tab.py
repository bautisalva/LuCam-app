from PyQt5.QtWidgets import QWidget, QHBoxLayout


class CaptureTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.addLayout(self.app.build_capture_left_layout())
        layout.addLayout(self.app.build_capture_controls_layout())
        self.setLayout(layout)
