import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QSlider, QLineEdit, QComboBox, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from skimage.filters import gaussian, threshold_otsu, sobel
from skimage.measure import find_contours
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks

class AnalysisTab(QWidget):
    def __init__(self, get_current_image_callback, log_callback):
        super().__init__()
        self.get_current_image = get_current_image_callback
        self.log = log_callback
        self.binary_result = None
        self.contours_result = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()  # Cambiado de QVBoxLayout a QHBoxLayout
    
        # === IZQUIERDA: Imagen ===
        self.image_label = QLabel("Imagen de análisis")
        self.image_label.setFixedSize(960 , 786)
        main_layout.addWidget(self.image_label)
    
        # === DERECHA: Controles en columna ===
        control_layout = QVBoxLayout()
    
        self.detect_button = QPushButton("Detectar Contorno")
        self.detect_button.clicked.connect(self.detect_contour)
        control_layout.addWidget(self.detect_button)
    
        self.method_selector = QComboBox()
        self.method_selector.addItems(["sobel", "binarizacion"])
        control_layout.addLayout(self._labeled_widget("Método:", self.method_selector))
    
        self.alpha_input = QLineEdit("0")
        control_layout.addLayout(self._labeled_widget("Alpha (fondo):", self.alpha_input))
    
        self.blur_input = QLineEdit("5")
        control_layout.addLayout(self._labeled_widget("Suavizado:", self.blur_input))
    
        self.percentil_input = QLineEdit("0")
        control_layout.addLayout(self._labeled_widget("% contorno mínimo:", self.percentil_input))
    
        self.save_bin_button = QPushButton("Guardar Imagen Binarizada")
        self.save_bin_button.clicked.connect(self.save_binary)
        control_layout.addWidget(self.save_bin_button)
    
        self.save_contours_button = QPushButton("Guardar Contornos como TXT")
        self.save_contours_button.clicked.connect(self.save_contours)
        control_layout.addWidget(self.save_contours_button)
    
        control_layout.addStretch()
        main_layout.addLayout(control_layout)
    
        self.setLayout(main_layout)


    def _labeled_widget(self, label, widget):
        container = QHBoxLayout()
        container.addWidget(QLabel(label))
        container.addWidget(widget)
        return container

    def detect_contour(self):
        image = self.get_current_image()
        if image is None:
            self.log("[ERROR] No hay imagen disponible para analizar.")
            return

        try:
            alpha = float(self.alpha_input.text())
            suavizado = int(self.blur_input.text())
            percentil = float(self.percentil_input.text())
            metodo = self.method_selector.currentText()

            # === PROCESAMIENTO DE IMAGEN ===
            corrected = image.astype(np.float32) - alpha * gaussian(image.astype(np.float32), sigma=100, preserve_range=True)
            centro, sigma, hist, top_peaks = self._detect_histogram_peaks(corrected)

            enhanced = np.exp(-0.5 * ((corrected - centro) / sigma) ** 2) * (corrected - centro)
            enhanced_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
            enhanced_uint16 = (enhanced_norm * 65535).astype(np.uint16)

            smooth = uniform_filter(enhanced_uint16, size=suavizado)
            centro1, sigma1, hist, top_peaks = self._detect_histogram_peaks(smooth)

            enhanced2 = 0.5 * (np.tanh(0.5 * ((smooth - centro1) / sigma1)) + 1)
            enhanced2_norm = (enhanced2 - enhanced2.min()) / (enhanced2.max() - enhanced2.min())
            enhanced2_uint16 = (enhanced2_norm * 65535).astype(np.uint16)

            threshold = threshold_otsu(enhanced2_uint16)
            binary = (enhanced2_uint16 > threshold).astype(np.uint16) * 65535

            if metodo == "sobel":
                sobel_image = sobel(enhanced2_uint16.astype(float) / 65535)
                contornos = self._find_contours_by_sobel(enhanced2_uint16, [0.16], percentil)
            else:
                contornos = self._find_large_contours(binary, percentil)

            self.binary_result = binary
            self.contours_result = contornos

            # Crear figura matplotlib sin mostrar
            fig, ax = plt.subplots(figsize=(9.6, 7.86), dpi=100)  # tamaño final: 400x300
            ax.imshow(binary, cmap='gray')
            
            for c in contornos:
                ax.plot(c[:, 1], c[:, 0], linewidth=0.5, color='cyan')
            
            ax.axis('off')
            fig.tight_layout(pad=0)
            
            # Convertir figura matplotlib a QPixmap
            canvas = FigureCanvas(fig)
            canvas.draw()
            width, height = canvas.get_width_height()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
            
            # Convertir a QImage
            qimage = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            
            self.image_label.setPixmap(pixmap)
            plt.close(fig)

            self.log(f"[INFO] Contornos detectados: {len(contornos)}")

        except Exception as e:
            self.log(f"[ERROR] Falló análisis de imagen: {e}")

    def save_binary(self):
        if self.binary_result is None:
            self.log("[ERROR] No hay imagen binarizada para guardar.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar binarizada", "", "PNG (*.png);;TIFF (*.tif)")
        if path:
            from skimage.io import imsave
            imsave(path, self.binary_result.astype(np.uint8))
            self.log(f"[INFO] Imagen binarizada guardada en {path}")

    def save_contours(self):
        if not self.contours_result:
            self.log("[ERROR] No hay contornos para guardar.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar contornos", "", "TXT (*.txt)")
        if path:
            with open(path, 'w') as f:
                for i, cont in enumerate(self.contours_result):
                    f.write(f"# Contorno {i}\n")
                    np.savetxt(f, cont, fmt="%.3f")
                    f.write("\n")
            self.log(f"[INFO] Contornos guardados en {path}")

    def _detect_histogram_peaks(self, image, min_intensity=5, min_dist=10):
        histograma, _ = np.histogram(image[image > min_intensity], bins=65535, range=(0, 65535))
        histograma[:5] = 0
        hist = gaussian(histograma.astype(float), sigma=5)
        peaks, _ = find_peaks(hist, distance=min_dist)
        peak_vals = hist[peaks]
        if len(peaks) < 2:
            raise ValueError("No se detectaron dos picos suficientemente separados en el histograma.")
        sorted_indices = np.argsort(peak_vals)[-2:]
        top_peaks = peaks[sorted_indices]
        top_peaks.sort()
        centro = (top_peaks[0] + top_peaks[1]) / 2
        sigma = abs(top_peaks[0] - centro)
        return centro, sigma, hist, top_peaks

    def _find_large_contours(self, binary, percentil_contornos=0):
        contours = find_contours(binary, level=0.5)
        if percentil_contornos > 0:
            def area_contorno(contour):
                x = contour[:, 1]
                y = contour[:, 0]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas = np.array([area_contorno(c) for c in contours])
            umbral = np.percentile(areas, percentil_contornos)
            return [c for c, a in zip(contours, areas) if a >= umbral]
        return contours

    def _find_contours_by_sobel(self, image, levels=[0.1], percentil_contornos=0):
        edges = sobel(image.astype(float) / 65535)
        contornos = []
        for nivel in levels:
            c = find_contours(edges, level=nivel)
            contornos.extend(c)
        if percentil_contornos > 0 and contornos:
            def area_contorno(contour):
                x = contour[:, 1]
                y = contour[:, 0]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas = np.array([area_contorno(c) for c in contornos])
            umbral = np.percentile(areas, percentil_contornos)
            contornos = [c for c, a in zip(contornos, areas) if a >= umbral]
        return contornos
