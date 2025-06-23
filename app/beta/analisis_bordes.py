import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_otsu, sobel
from skimage.measure import find_contours
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks

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