import os
import datetime
import numpy as np
import json
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QComboBox, QFileDialog, QPlainTextEdit, QGridLayout
)
from PyQt5.QtCore import Qt
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import resize
from PyQt5.QtGui import QImage, QPixmap
from analisis_bordes import ImageEnhancer


class AnalysisTab(QWidget):
    def __init__(self, parent=None, get_image_callback=None):
        super().__init__(parent)
        self.get_image_callback = get_image_callback
        self.loaded_image = None
        self.binary_result = None
        self.contours_result = None
        self.analysis_images = []
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        control_layout = QHBoxLayout()
        self.suavizado_spin = QSpinBox()
        self.suavizado_spin.setRange(1, 20)
        self.suavizado_spin.setValue(3)
        control_layout.addWidget(QLabel("Suavizado:"))
        control_layout.addWidget(self.suavizado_spin)

        self.percentil_spin = QSpinBox()
        self.percentil_spin.setRange(0, 100)
        self.percentil_spin.setValue(99)
        control_layout.addWidget(QLabel("% Contornos:"))
        control_layout.addWidget(self.percentil_spin)

        self.distpico_spin = QSpinBox()
        self.distpico_spin.setRange(1, 100)
        self.distpico_spin.setValue(10)
        control_layout.addWidget(QLabel("Dist. mín. picos:"))
        control_layout.addWidget(self.distpico_spin)

        self.metodo_combo = QComboBox()
        self.metodo_combo.addItems(["sobel", "binarizacion"])
        control_layout.addWidget(QLabel("Método:"))
        control_layout.addWidget(self.metodo_combo)

        main_layout.addLayout(control_layout)

        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Aplicar análisis de bordes")
        self.run_button.clicked.connect(self.run_analysis)
        button_layout.addWidget(self.run_button)

        self.load_button = QPushButton("Cargar imagen desde archivo")
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)

        self.save_button = QPushButton("Guardar resultados")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        main_layout.addLayout(button_layout)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFixedHeight(100)
        main_layout.addWidget(self.console)

        # Display area for images
        self.image_grid = QGridLayout()
        self.image_labels = []
        for i in range(6):
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(300, 220)
            label.setStyleSheet("border: 1px solid gray")
            self.image_labels.append(label)
            self.image_grid.addWidget(label, i // 3, i % 3)
        main_layout.addLayout(self.image_grid)

        self.setLayout(main_layout)

    def log(self, text):
        self.console.appendPlainText(text)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Imágenes (*.tif *.tiff *.png *.jpg)")
        if file_path:
            try:
                image = imread(file_path)
                if len(image.shape) == 3:
                    image = (rgb2gray(image) * 255).astype(np.uint8)
                else:
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                self.loaded_image = image
                self.log(f"[OK] Imagen cargada: {file_path}")
            except Exception as e:
                self.log(f"[ERROR] No se pudo cargar la imagen: {e}")

    def run_analysis(self):
        image = self.loaded_image if self.loaded_image is not None else (
            self.get_image_callback() if self.get_image_callback else None
        )

        if image is None:
            self.log("[ERROR] No hay imagen cargada ni capturada.")
            return

        if len(image.shape) == 3:
            image = (rgb2gray(image) * 255).astype(np.uint8)

        try:
            enhancer = ImageEnhancer(image)
            binary, contornos, etapas, hist_img = enhancer.procesar_gui(
                suavizado=self.suavizado_spin.value(),
                percentil_contornos=self.percentil_spin.value(),
                min_dist_picos=self.distpico_spin.value(),
                metodo_contorno=self.metodo_combo.currentText(),
                mostrar=False,
                retornar_etapas=True
            )

            self.binary_result = binary
            self.contours_result = contornos
            self.save_button.setEnabled(True)
            self.log(f"[OK] Se detectaron {len(contornos)} contornos.")

            for label, etapa in zip(self.image_labels[:5], etapas):
                self._mostrar_imagen_en_label(etapa, label)
            self._mostrar_imagen_en_label(hist_img, self.image_labels[5], is_rgb=True)

        except Exception as e:
            self.log(f"[ERROR] Falló el análisis: {e}")

    def _mostrar_imagen_en_label(self, image, label, is_rgb=False):
        image = resize(image, (220, 300), preserve_range=True).astype(np.uint8)
        if is_rgb:
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        else:
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)

    def save_results(self):
        if self.binary_result is None or self.contours_result is None:
            self.log("[ERROR] No hay resultados para guardar.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de destino")
        if not folder:
            return

        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            bin_path = os.path.join(folder, f'binarizada_{timestamp}.tif')
            cont_path = os.path.join(folder, f'contornos_{timestamp}.json')

            imsave(bin_path, self.binary_result.astype(np.uint8))
            contornos_serializables = [c.tolist() for c in self.contours_result]
            with open(cont_path, 'w') as f:
                json.dump(contornos_serializables, f)

            self.log(f"[OK] Imagen binarizada guardada en: {bin_path}")
            self.log(f"[OK] Contornos guardados en: {cont_path}")

        except Exception as e:
            self.log(f"[ERROR] No se pudieron guardar los resultados: {e}")
