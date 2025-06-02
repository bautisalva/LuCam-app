from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QComboBox,
    QPushButton, QSlider, QLineEdit
)
from PyQt5.QtCore import Qt


class PreviewTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()

        # Vista y consola
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.app.preview_label_preview)
        left_layout.addWidget(self.app.console_preview)
        layout.addLayout(left_layout)

        controls_layout = QVBoxLayout()
        self.app.sliders = {}
        self.app.inputs = {}

        # Selector de FPS
        fps_group = QGroupBox("FPS")
        fps_layout = QHBoxLayout()
        self.app.fps_selector = QComboBox()
        for fps in self.app.available_fps:
            self.app.fps_selector.addItem(f"{fps:.2f}")
        self.app.fps_selector.currentTextChanged.connect(self.app.change_fps)
        fps_layout.addWidget(QLabel("Frames por segundo:"))
        fps_layout.addWidget(self.app.fps_selector)
        fps_group.setLayout(fps_layout)
        controls_layout.addWidget(fps_group)

        try:
            current_fps = self.app.camera.GetFormat()[1]
            index = self.app.fps_selector.findText(f"{current_fps:.2f}")
            if index != -1:
                self.app.fps_selector.setCurrentIndex(index)
        except Exception as e:
            self.app.log_message(f"[WARNING] No se pudo establecer FPS actual: {e}")

        # Sliders de propiedades de cámara
        for prop, (min_val, max_val, default) in self.app.properties.items():
            group = QGroupBox(prop.capitalize())
            group_layout = QHBoxLayout()

            label = QLabel(f"{default}")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * 100))
            slider.setMaximum(int(max_val * 100))
            slider.setValue(int(default * 100))
            slider.valueChanged.connect(lambda value, p=prop: self.app.update_property(p, value / 100))

            input_field = QLineEdit(str(default))
            input_field.setFixedWidth(50)
            input_field.editingFinished.connect(lambda p=prop, field=input_field: self.app.set_property_from_input(p, field))

            group_layout.addWidget(label)
            group_layout.addWidget(slider)
            group_layout.addWidget(input_field)
            group.setLayout(group_layout)

            controls_layout.addWidget(group)
            self.app.sliders[prop] = slider
            self.app.inputs[prop] = input_field

        # Botones
        save_btn = QPushButton("Guardar Parámetros de Preview")
        save_btn.clicked.connect(self.app.save_preview_parameters)
        controls_layout.addWidget(save_btn)

        load_btn = QPushButton("Cargar Parámetros")
        load_btn.clicked.connect(self.app.load_parameters)
        controls_layout.addWidget(load_btn)

        refresh_btn = QPushButton("Refrescar desde Cámara")
        refresh_btn.clicked.connect(self.app.apply_real_camera_values_to_sliders)
        controls_layout.addWidget(refresh_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        self.setLayout(layout)

        self.app.apply_real_camera_values_to_sliders()

