import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import find_contours
from scipy.ndimage import uniform_filter

class ImageEnhancer:
    def __init__(self, imagen, sigma_background=1000, alpha=1):
        self.image = imagen
        self.sigma_background = sigma_background
        self.alpha = alpha

    def _subtract_background(self):
        background = gaussian(self.image.astype(np.float32), sigma=self.sigma_background, preserve_range=True)
        corrected = self.image.astype(np.float32) - self.alpha * background
        return np.clip(corrected, 0, None)

    def _enhance(self, corrected, metodo="chi2", k=4, escala=40, ganancia1=0.1, ganancia2=0.01):
        if metodo == "chi2":
            delta = np.clip(corrected - np.mean(corrected), 1e-3, None)
            delta_rescaled = delta / escala
            enhanced = (delta_rescaled ** (k / 2 - 1)) * np.exp(-delta_rescaled / 2)
        elif metodo == "gaussiana":
            delta = corrected - 255/2 + np.mean(corrected)
            ganancia = 2 * np.std(corrected)
            enhanced = np.exp(-(delta / ganancia) ** 2)
        elif metodo == "tanh_diff":
            enhanced = self._enhance_tanh_diff(corrected, ganancia1=ganancia1, ganancia2=ganancia2)
        elif metodo == "tanh_diff2":
            enhanced = self._enhance_tanh_diff2(corrected, ganancia1=ganancia1, ganancia2=ganancia2)
        else:
            raise ValueError("Método inválido. Usar 'chi2', 'gaussiana' o 'tanh_diff'.")
        return enhanced

    def _enhance_tanh_diff(self, corrected, ganancia1=0.1, ganancia2=0.01):
        delta = corrected - np.mean(corrected)
        return 0.5 * (np.tanh(ganancia1 * delta) - np.tanh(ganancia2 * delta) + 1)
    
    def _enhance_tanh_diff2(self, corrected, ganancia1=0.1, ganancia2=0.01):
        delta = corrected - 255/2
        return np.exp(-(delta / 400) ** 2) * 0.5 * (np.tanh(ganancia1 * delta) - np.tanh(ganancia2 * delta) + 1)

    def _apply_tanh(self, image, ganancia=0.1):
        delta = image - np.mean(image)
        return 0.5 * (np.tanh(ganancia * delta) + 1)

    def _find_large_contours(self, binary, percentil=99.9):
        contours = find_contours(binary, level=0.5)

        def area_contorno(contour):
            x = contour[:, 1]
            y = contour[:, 0]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        areas = np.array([area_contorno(c) for c in contours])
        umbral = np.percentile(areas, percentil)
        return [c for c, a in zip(contours, areas) if a >= umbral]

    def procesar(self, metodo="chi2", k=4, escala=5, suavizado=2, ganancia_tanh=0.1, ganancia1=0.1, ganancia2=0.01, mostrar=True):
        corrected = self._subtract_background()
        enhanced = self._enhance(corrected, metodo=metodo, k=k, escala=escala, ganancia1=ganancia1, ganancia2=ganancia2)
        enhanced_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
        enhanced_uint8 = (enhanced_norm * 255).astype(np.uint8)

        smooth = uniform_filter(enhanced_uint8, size=suavizado)

        enhanced2 = self._apply_tanh(smooth, ganancia=ganancia_tanh)
        enhanced2_norm = (enhanced2 - enhanced2.min()) / (enhanced2.max() - enhanced2.min())
        enhanced2_uint8 = (enhanced2_norm * 255).astype(np.uint8)

        threshold = threshold_otsu(enhanced2_uint8)
        binary = (enhanced2_uint8 > threshold).astype(np.uint8) * 255

        contornos = self._find_large_contours(binary)

        if mostrar:
            self._mostrar_resultados(corrected, enhanced_uint8, smooth, enhanced2_uint8, binary, contornos, metodo, k)

        return binary, contornos

    def _mostrar_resultados(self, corrected, enhanced_uint8, smooth, enhanced2_uint8, binary, contornos, metodo, k):
        if metodo == "chi2":
            titulo = f"Chi² (k={k})"
        elif metodo == "gaussiana":
            titulo = "Gaussiana"
        elif metodo == "tanh_diff":
            titulo = "Tanh diferencia"
        else:
            titulo = metodo

        plt.figure(figsize=(18, 12))

        plt.subplot(2, 3, 1)
        plt.imshow(self.image, cmap='gray')
        for c in contornos:
            plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
        plt.title("Original + contornos (píxeles)")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(corrected, cmap='gray')
        plt.title("Con fondo restado")
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(enhanced_uint8, cmap='gray')
        plt.title(titulo)
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(smooth, cmap='gray')
        plt.title("Suavizado (media móvil)")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(enhanced2_uint8, cmap='gray')
        plt.title("Tanh")
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(binary, cmap='gray')
        for c in contornos:
            plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
        plt.title(f"Contornos ({len(contornos)} detectados)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

#%%

from skimage.io import imread

imagen = imread("C:/Users/Tomas/Desktop/FACULTAD/LABO 6/Resta-P8139-150Oe-50ms-1000.tif")[400:700, 475:825]  # lo recortás vos
enhancer = ImageEnhancer(imagen=imagen)
binary, contornos = enhancer.procesar(metodo="tanh_diff", k=4 , escala=40 , ganancia1=0.1, ganancia2=0.01, suavizado=6)
