import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_otsu, sobel
from skimage.measure import find_contours
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSpinBox, QComboBox, QLabel, QPushButton, QFileDialog, QGridLayout, QPlainTextEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import os
import json
import datetime
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np



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
                    image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
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
        if not is_rgb:
            image = resize(image, (220, 300), preserve_range=True).astype(np.uint8)
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Grayscale8)
        else:
            qimage = QImage(image.data, image.shape[1]*3, image.shape[0], QImage.Format_RGB888)
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


class ImageEnhancer:
    def __init__(self, imagen, sigma_background=100, alpha=0):
        self.image = imagen
        self.sigma_background = sigma_background
        self.alpha = alpha

    def _subtract_background(self):
        background = gaussian(self.image.astype(np.float32), sigma=self.sigma_background, preserve_range=True)
        corrected = self.image.astype(np.float32) - self.alpha * background
        return corrected

    def _detect_histogram_peaks(self, image, min_intensity=5, min_dist=30):
        histograma, bins = np.histogram(image[image > min_intensity], bins=256, range=(0, 255))
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

    def _enhance_tanh_diff2(self, corrected, centro, sigma):
        delta = corrected - centro
        return np.exp(-0.5 * (delta / sigma) ** 2) * delta

    def _apply_tanh(self, image, ganancia=1, centro=100, sigma=50):
        delta = image - centro
        return 0.5 * (np.tanh(0.5 * delta / sigma) + 1)

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
        edges = sobel(image.astype(float) / 255.0)
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

    def procesar_gui(self, suavizado=5, ganancia_tanh=0.1, mostrar=False, percentil_contornos=0,
                     min_dist_picos=30, metodo_contorno="sobel", retornar_etapas=False):
        corrected = self._subtract_background()
        centro, sigma, hist, top_peaks = self._detect_histogram_peaks(corrected, min_dist=min_dist_picos)
    
        # Realce con función gaussiana x*exp(-(x/sigma)^2)
        enhanced = self._enhance_tanh_diff2(corrected, centro, sigma)
        enhanced_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
        enhanced_uint8 = (enhanced_norm * 255).astype(np.uint8)
    
        # Suavizado y segunda realce tipo tanh
        smooth = uniform_filter(enhanced_uint8, size=suavizado)
        centro1, sigma1, hist2, top_peaks2 = self._detect_histogram_peaks(smooth, min_dist=min_dist_picos)
    
        enhanced2 = self._apply_tanh(smooth, ganancia=ganancia_tanh, centro=centro1, sigma=sigma1)
        enhanced2_norm = (enhanced2 - enhanced2.min()) / (enhanced2.max() - enhanced2.min())
        enhanced2_uint8 = (enhanced2_norm * 255).astype(np.uint8)
    
        # Binarización
        threshold = np.mean(enhanced2_uint8)
        binary = (enhanced2_uint8 > threshold).astype(np.uint8) * 255
    
        # Detección de contornos
        if metodo_contorno == "sobel":
            contornos = self._find_contours_by_sobel(enhanced2_uint8, levels=[0.16], percentil_contornos=percentil_contornos)
        elif metodo_contorno == "binarizacion":
            contornos = self._find_large_contours(binary, percentil_contornos=percentil_contornos)
        else:
            raise ValueError(f"Método de contorno no reconocido: {metodo_contorno}")
    
        # Etapas para mostrar en GUI
        imagen_original = self.image
        imagen_realzada = enhanced_uint8
        imagen_suavizada = smooth
        imagen_tanh = enhanced2_uint8
        imagen_binaria = binary
    
        # Histograma como imagen
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.plot(hist2, color='black')
        ax.scatter(top_peaks2, hist2[top_peaks2], color='red')
        ax.set_title("Histograma de Intensidades")
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Frecuencia")
        fig.tight_layout()
    
        # Convertir figura matplotlib a QImage
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        hist_img = Image.open(buf).convert("RGB")
        hist_img = hist_img.resize((300, 220))
        hist_np = np.array(hist_img)
        buf.close()
        plt.close(fig)
    
        if retornar_etapas:
            return binary, contornos, [imagen_original, imagen_realzada, imagen_suavizada,
                                       imagen_tanh, imagen_binaria], hist_np
        else:
            return binary, contornos


    def _mostrar_resultados(self, enhanced_uint8, smooth, enhanced2_uint8, binary, contornos, hist, top_peaks, threshold, imagen_contorno):
        plt.figure(figsize=(18, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(self.image, cmap='gray')
        for c in contornos:
            plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
        plt.title("Original + contornos")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(enhanced_uint8, cmap='gray')
        plt.title("Realce (x*exp(-0.5(x/sigma)**2))")
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(smooth, cmap='gray')
        plt.title("Suavizado")
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(enhanced2_uint8, cmap='gray')
        plt.title("Tanh final")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(imagen_contorno, cmap='gray')
        for c in contornos:
            plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
        plt.title(f'Contorno sobre {"sobel" if imagen_contorno is enhanced2_uint8 else "binarizada"}')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.plot(hist, color='gray')
        plt.scatter(top_peaks, hist[top_peaks], color='red')
        plt.title("Histograma + Picos seleccionados")
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")

        print(f"Cantidad de contornos detectados: {len(contornos)}")

        plt.tight_layout()
        plt.show()