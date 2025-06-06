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

    def procesar(self, suavizado=5, ganancia_tanh=0.1, mostrar=True, percentil_contornos=0, min_dist_picos=30, metodo_contorno="sobel"):
        corrected = self._subtract_background()
        centro, sigma, hist, top_peaks = self._detect_histogram_peaks(corrected, min_dist=min_dist_picos)

        enhanced = self._enhance_tanh_diff2(corrected, centro, sigma)
        enhanced_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
        enhanced_uint8 = (enhanced_norm * 255).astype(np.uint8)

        smooth = uniform_filter(enhanced_uint8, size=suavizado)
        centro1, sigma1, hist, top_peaks = self._detect_histogram_peaks(smooth, min_dist=min_dist_picos)

        enhanced2 = self._apply_tanh(smooth, ganancia=ganancia_tanh, centro=centro1, sigma=sigma1)
        enhanced2_norm = (enhanced2 - enhanced2.min()) / (enhanced2.max() - enhanced2.min())
        enhanced2_uint8 = (enhanced2_norm * 255).astype(np.uint8)

        #threshold = threshold_otsu(enhanced2_uint8)
        threshold = np.mean(enhanced2_uint8)
        print(threshold)
        binary = (enhanced2_uint8 > threshold).astype(np.uint8) * 255

        if metodo_contorno == "sobel":
            sobel_image = sobel(enhanced2_uint8.astype(float) / 255.0)
            contornos = self._find_contours_by_sobel(enhanced2_uint8, levels=[0.16], percentil_contornos=percentil_contornos)
            imagen_contorno = sobel_image  # Ahora sí, se muestra la imagen del sobel real
        elif metodo_contorno == "binarizacion":
            contornos = self._find_large_contours(binary, percentil_contornos=percentil_contornos)
            imagen_contorno = binary
        else:
            raise ValueError(f"Método de contorno no reconocido: {metodo_contorno}")

        if mostrar:
            self._mostrar_resultados(enhanced_uint8, smooth, enhanced2_uint8, binary, contornos, hist, top_peaks, threshold, imagen_contorno)

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



#%%

from skimage.io import imread

imagen = imread(r"C:\Users\Marina\Documents\Labo 6\LuCam-app\test\test fotos\Resta-P8139-150Oe-50ms-1000.tif")[400:700, 475:825]
#imagen = imread("C:/Users/Tomas/Desktop/FACULTAD/LABO 6/Resta-P8139-150Oe-50ms-1000.tif")[400:700, 475:825]
enhancer = ImageEnhancer(imagen=imagen)
binary, contornos = enhancer.procesar(
    suavizado=3,
    percentil_contornos=99.9,
    min_dist_picos=5,
    metodo_contorno="binarizacion"
)

#%%

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from imageio.v2 import imread

# Cargar y recortar imagen
#imagen = imread("C:/Users/Tomas/Desktop/FACULTAD/LABO 6/Resta-P8139-150Oe-50ms-1000.tif")[400:700, 475:825]


#background = gaussian(im.astype(np.float32), sigma=1000, preserve_range=True)
#corrected = im.astype(np.float32) - background
#corrected_norm = (corrected - corrected.min()) / (corrected.max() - corrected.min())
#imagen = (corrected_norm * 255).astype(np.uint8)

fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar la imagen
ax_img.imshow(imagen, cmap='gray')
ax_img.set_title("Seleccioná una zona")
ax_img.axis('off')

# Histograma vacío al inicio
hist_plot = ax_hist.hist([], bins=256, color='gray')
ax_hist.set_title("Histograma de la zona")
ax_hist.set_xlabel("Intensidad")
ax_hist.set_ylabel("Frecuencia")

# Función que se ejecuta cuando se selecciona una zona
def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    zona = imagen[ymin:ymax, xmin:xmax]
    
    ax_hist.cla()  # limpiar histograma anterior
    ax_hist.hist(zona.ravel(), bins=256, color='gray')
    ax_hist.set_title(f"Histograma zona ({xmin}:{xmax}, {ymin}:{ymax})")
    ax_hist.set_xlabel("Intensidad")
    ax_hist.set_ylabel("Frecuencia")
    
    fig.canvas.draw_idle()

# Activar el selector
selector = RectangleSelector(ax_img, onselect,
                             useblit=True,
                             button=[1],
                             minspanx=5, minspany=5,
                             spancoords='pixels',
                             interactive=True)

plt.tight_layout()
plt.show()

#%%


x = np.linspace(0,255,1000)

y = (x-79)*(np.tanh((x-27)/(27-12))-np.tanh((x-131)/(131-117)))

plt.plot(x,y)


