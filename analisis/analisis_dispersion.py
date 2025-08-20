# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:37:08 2025

@author: Toto y bauti
"""

import numpy as np
from skimage import measure
from scipy.spatial import cKDTree

def var_u(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calcula Var(u) entre dos imágenes binarias de dominios magnéticos consecutivas.
    
    Parámetros
    ----------
    img1 : np.ndarray
        Imagen binaria en t (0 = fuera del dominio, 1 = dentro del dominio).
    img2 : np.ndarray
        Imagen binaria en t+1.
    
    Retorna
    -------
    var_u : float
        Varianza del desplazamiento Var(u).
    """

    # 1. Diferencia de pertenencia Δa
    delta_a = img2.astype(int) - img1.astype(int)
    
    # 2. Extraer contorno del dominio en img1
    contours = measure.find_contours(img1, level=0.5)
    # Tomamos el contorno más largo como el principal
    contour = max(contours, key=len)
    
    # Perímetro (P): longitud del borde
    diffs = np.diff(contour, axis=0)
    P = np.sum(np.sqrt((diffs**2).sum(axis=1)))
    
    # 3. Píxeles que cambiaron de estado
    changed_pixels = np.argwhere(delta_a != 0)
    if len(changed_pixels) == 0:
        return 0.0  # no hubo cambio
    
    # 4. Construir KDTree del contorno para medir distancias u'_i
    tree = cKDTree(contour)
    distances, _ = tree.query(changed_pixels)
    
    # 5. Cálculo de Var(u) según ecuación (16)
    sum_uprime = np.sum(np.abs(distances))
    sum_abs_da = np.sum(delta_a)
    
    var_u_value = (2 * sum_uprime / P) - (sum_abs_da / P) ** 2
    
    return var_u_value


# Ejemplo de uso con dos imágenes binarias (numpy arrays)
from skimage.draw import disk

img1 = np.zeros((100,100), dtype=int)
rr, cc = disk((50,50), 20)
img1[rr, cc] = 1

img2 = np.zeros((100,100), dtype=int)
rr, cc = disk((50,50), 22)  # dominio crece un poco
img2[rr, cc] = 1

valor = var_u(img1, img2)
print("Var(u) =", valor)

#%%

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.spatial import cKDTree

def var_u_with_plot(img1: np.ndarray, img2: np.ndarray, n_show: int = 30) -> float:
    """
    Calcula Var(u) y muestra una visualización de los píxeles cambiados
    y sus distancias al borde.
    
    Parámetros
    ----------
    img1, img2 : np.ndarray
        Imágenes binarias consecutivas (0 = fuera, 1 = dentro).
    n_show : int
        Cantidad de píxeles cambiados a los que se les dibuja la línea de distancia.
    
    Retorna
    -------
    var_u : float
        Varianza del desplazamiento Var(u).
    """
    
    delta_a = img2.astype(int) - img1.astype(int)
    
    # Contorno de referencia (C1)
    contours = measure.find_contours(img1, level=0.5)
    contour = max(contours, key=len)
    
    # Perímetro (P)
    diffs = np.diff(contour, axis=0)
    P = np.sum(np.sqrt((diffs**2).sum(axis=1)))
    
    # Píxeles cambiados
    changed_pixels = np.argwhere(delta_a != 0)
    if len(changed_pixels) == 0:
        print("No hubo cambio entre imágenes")
        return 0.0
    
    # Distancias a C1
    tree = cKDTree(contour)
    distances, idx = tree.query(changed_pixels)
    
    sum_uprime = np.sum(distances)
    sum_abs_da = np.sum(np.abs(delta_a))
    
    var_u_value = (2 * sum_uprime / P) - (sum_abs_da / P) ** 2
    
    # --- Visualización ---
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img2, cmap="gray")
    
    # Contorno C1 en rojo
    ax.plot(contour[:,1], contour[:,0], 'r-', lw=1.5, label="Contorno C1")
    
    # Píxeles cambiados en azul
    ax.plot(changed_pixels[:,1], changed_pixels[:,0], 'bo', ms=2, label="Δa ≠ 0")
    
    # Dibujar algunas líneas de distancia
    for i in np.random.choice(len(changed_pixels), size=min(n_show, len(changed_pixels)), replace=False):
        px = changed_pixels[i]
        nearest = contour[idx[i]]
        ax.plot([px[1], nearest[1]], [px[0], nearest[0]], 'g-', lw=0.8, alpha=0.6)
    
    ax.legend()
    ax.set_title(f"Var(u) = {var_u_value:.4f}")
    plt.show()
    
    return var_u_value

from skimage.draw import disk

# Imagen inicial
img1 = np.zeros((100,100), dtype=int)
rr, cc = disk((50,50), 20)
img1[rr, cc] = 1

# Imagen posterior (dominio crece un poco)
img2 = np.zeros((100,100), dtype=int)
rr, cc = disk((50,50), 22)
img2[rr, cc] = 1

valor = var_u_with_plot(img1, img2)
print("Var(u) =", valor)

#%%
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure
from scipy.spatial import cKDTree

# ==============================
# Función de cálculo Var(u)
# ==============================
def var_u(img1: np.ndarray, img2: np.ndarray) -> float:
    delta_a = img2.astype(int) - img1.astype(int)

    # Contorno de referencia (C1)
    contours = measure.find_contours(img1, level=0.5)
    if len(contours) == 0:
        return 0.0
    contour = max(contours, key=len)

    # Perímetro P
    diffs = np.diff(contour, axis=0)
    P = np.sum(np.sqrt((diffs**2).sum(axis=1)))
    if P == 0:
        return 0.0

    # Píxeles cambiados
    changed_pixels = np.argwhere(delta_a != 0)
    if len(changed_pixels) == 0:
        return 0.0

    # Distancias a C1
    tree = cKDTree(contour)
    distances, _ = tree.query(changed_pixels)

    sum_uprime = np.sum(distances)
    sum_abs_da = np.sum(np.abs(delta_a))

    var_u_value = (2 * sum_uprime / P) - (sum_abs_da / P) ** 2
    return var_u_value


# ==============================
# Main
# ==============================
# Ruta de imágenes
folder = r"C:\Users\Marina\Documents\Labo 6\imagenes\test_velocidades"

# Regex para extraer el número al final del nombre
pattern = re.compile(r".*-(\d+)$")

# Listar y ordenar imágenes
files = []
for f in os.listdir(folder):
    match = pattern.match(os.path.splitext(f)[0])  # sin extensión
    if match:
        step = int(match.group(1))
        files.append((step, os.path.join(folder, f)))

# Ordenar por step
files.sort(key=lambda x: x[0])

# Leer imágenes binarias
images = [io.imread(path) > 0 for _, path in files]  # umbral simple

# Calcular Var(u) en cada paso consecutivo
var_values = []
steps = []

for k in range(len(images) - 1):
    v = var_u(images[k], images[k+1])
    var_values.append(v)
    steps.append(k)  # paso k → k+1

# ==============================
# Gráfico
# ==============================
plt.figure(figsize=(8,5))
plt.plot(steps, var_values, marker='o', lw=1.5)
plt.xlabel("Paso (n → n+1)")
plt.ylabel("Var(u)")
plt.title("Variación de Var(u) entre pasos consecutivos")
plt.grid(True)
plt.show()

#%%

from desplazamiento_area_perimetro import ContourAnalysis
import pandas as pd

RAIZ = r'C:\Users\Marina\Documents\Labo 6\imagenes'
CAMPOS = {
    80: {
        "folder": "250 x 10",
        "delta_t": 0.010,
        "start": 15,
        "end": None,
        "crop": (slice(140, 280), slice(345, 495))
        }
}
ESCALA = 0.36 / 1e6  # um/pixel
PATRON = r'resta_(\d{8}_\d{6})\.tif'

images = []

for campo, info in CAMPOS.items():
    print(f"Procesando campo: {campo} Oe")
    folder = os.path.join(RAIZ, info['folder'])
    output = os.path.join(folder, "analisis")
    
    analyzer = ContourAnalysis(
        image_dir=folder,
        crop_region=info['crop'],
        output_dir=output,
        start_from=info['start'],
        already_binarized=False,
        processing_params={
            'suavizado': 3,
            'percentil_contornos': 99.99,
            'min_dist_picos': 8000,
            'metodo_contorno': "binarizacion"
        },
        filename_pattern=PATRON
    )

    analyzer.process_images()
    results = analyzer.results

    filenames = sorted([f for f in results if 'error' not in results[f]])
    if info['end'] is not None:
        filenames = filenames[:info['end']]

    for i in range(len(filenames) - 1):
        images.append(results[filenames[i]]['binary'])

# Leer imágenes binarias
#images = [io.imread(path) > 0 for _, path in files]  # umbral simple

# Calcular Var(u) en cada paso consecutivo
var_values = []
steps = []

for k in range(len(images) - 1):
    v = var_u(images[k], images[k+1])
    var_values.append(v)
    steps.append(k)  # paso k → k+1

# ==============================
# Gráfico
# ==============================
plt.figure(figsize=(8,5))
plt.plot(steps, var_values, marker='o', lw=1.5)
plt.xlabel("Paso (n → n+1)")
plt.ylabel("Var(u)")
plt.title("Variación de Var(u) entre pasos consecutivos")
plt.grid(True)
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_otsu, sobel
from skimage.measure import find_contours
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

class ImageEnhancer:
    def __init__(self, imagen, sigma_background=100, alpha=0):
        self.image = imagen
        self.sigma_background = sigma_background
        self.alpha = alpha

    def _subtract_background(self):
        background = gaussian(self.image.astype(np.float32), sigma=self.sigma_background, preserve_range=True)
        corrected = self.image.astype(np.float32) - self.alpha * background
        return corrected

    def _detect_histogram_peaks(self, image, min_intensity=5, min_dist=30, usar_dos_picos=True):
        histograma, bins = np.histogram(image[image > min_intensity], bins=32767*2, range=(0, 32767*2))
        histograma[:5] = 0
        hist = gaussian(histograma.astype(float), sigma=650)
        peaks, _ = find_peaks(hist, distance=min_dist)
        peak_vals = hist[peaks]

        if len(peaks) >= 2 and usar_dos_picos:
            sorted_indices = np.argsort(peak_vals)[-2:]
            top_peaks = peaks[sorted_indices]
            top_peaks.sort()
            centro = (top_peaks[0] + top_peaks[1]) / 2
            sigma = abs(top_peaks[0] - centro)
        elif len(peaks) >= 1:
            mu = peaks[np.argmax(peak_vals)]
            def gauss(x, A, sigma):
                return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

            x_data = np.linspace(0, 32767*2, 32767*2)
            y_data = hist
            try:
                popt, _ = curve_fit(gauss, x_data, y_data, p0=[hist[mu], 10])
                A, sigma = popt
                centro = mu
            except RuntimeError:
                centro = mu
                sigma = 10
            top_peaks = np.array([mu])
        else:
            raise ValueError("No se detectaron picos en el histograma.")

        return centro, sigma, hist, top_peaks

    def _enhance_tanh_diff2(self, corrected, centro, sigma):
        delta = corrected - centro
        return np.exp(-0.5 * (delta / sigma) ** 2) * delta

    def _apply_tanh(self, image, ganancia=1, centro=100, sigma=50):
        delta = image - centro
        return -0.5 * (np.tanh(0.5 * delta / sigma) + 1)

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
        edges = sobel(image.astype(float) / 65534)
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

    def procesar(self, suavizado=5, ganancia_tanh=0.1, mostrar=True,
                 percentil_contornos=0, min_dist_picos=30,
                 metodo_contorno="sobel", usar_dos_picos=True):

        corrected = self._subtract_background()
        centro, sigma, hist, top_peaks = self._detect_histogram_peaks(corrected, min_dist=min_dist_picos, usar_dos_picos=True)

        enhanced = self._enhance_tanh_diff2(corrected, centro, sigma)
        enhanced_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
        enhanced_uint8 = (enhanced_norm * 65534).astype(np.uint16)

        smooth = uniform_filter(enhanced_uint8, size=suavizado)
        centro1, sigma1, hist1, top_peaks1 = self._detect_histogram_peaks(smooth, min_dist=min_dist_picos, usar_dos_picos=True)

        enhanced2 = self._apply_tanh(smooth, ganancia=ganancia_tanh, centro=centro1, sigma=sigma1)
        enhanced2_norm = (enhanced2 - enhanced2.min()) / (enhanced2.max() - enhanced2.min())
        enhanced2_uint8 = (enhanced2_norm * 65534).astype(np.uint16)

        threshold = threshold_otsu(enhanced2_uint8)
        print(threshold)
        binary = (enhanced2_uint8 > threshold).astype(np.uint16) * 65534

        if metodo_contorno == "sobel":
            sobel_image = sobel(enhanced2_uint8.astype(float) / 65534)
            contornos = self._find_contours_by_sobel(enhanced2_uint8, levels=[0.16], percentil_contornos=percentil_contornos)
            imagen_contorno = sobel_image
        elif metodo_contorno == "binarizacion":
            contornos = self._find_large_contours(binary, percentil_contornos=percentil_contornos)
            imagen_contorno = binary
        else:
            raise ValueError(f"Método de contorno no reconocido: {metodo_contorno}")

        return binary, contornos, hist
    
#%%
from skimage.io import imread, imsave

RAIZ = r'C:\Users\Marina\Documents\Labo 6\imagenes\250 x 10'
PATRON = r'resta_(\d{8}_\d{6})\.tif'

def load_images(PATRON,RAIZ):
    pattern = PATRON
    files = []

    for f in os.listdir(RAIZ):
        match = re.match(pattern, f)
        if match:
            key = match.group(1)

            # Intentar detectar si key es un número
            if re.fullmatch(r'\d+', key):
                sort_key = int(key)
            elif re.fullmatch(r'\d{8}_\d{6}', key):  # timestamp tipo 20250704_101230
                sort_key = key
            else:
                # Último recurso: buscar número al final
                num_match = re.search(r'(\d+)(?=\.\w+$)', f)
                sort_key = int(num_match.group(1)) if num_match else f

            files.append((sort_key, f))

    files.sort(key=lambda x: x[0])
    files = files[15:]

    images, filenames = [], []
    for _, fname in files:
        try:
            img = imread(os.path.join(RAIZ, fname))[140:280,345:495] 
            images.append(img)
            filenames.append(fname)
        except Exception as e:
            print(f"Error cargando {fname}: {str(e)}")

    return images, filenames

imag,_ = load_images(PATRON,RAIZ)

img = []

for i in range(len(imag)):
    enhancer = ImageEnhancer(imagen=imag[i])
    binary = enhancer.procesar(
        suavizado=3,
        percentil_contornos=99.9,
        min_dist_picos=8000,
        metodo_contorno="binarizacion"
    )
    
    img.append(binary)
    
plt.imshow(img[16])
    
#%%    

# Calcular Var(u) en cada paso consecutivo
var_values = []
steps = []

for k in range(len(img) - 1):
    v = var_u(img[k], img[k+1])
    var_values.append(v)
    steps.append(k)  # paso k → k+1

# ==============================
# Gráfico
# ==============================
plt.figure(figsize=(8,5))
plt.plot(steps, var_values, marker='o', lw=1.5)
plt.xlabel("Paso (n → n+1)")
plt.ylabel("Var(u)")
plt.ylim(-10,10)
plt.title("Variación de Var(u) entre pasos consecutivos")
plt.grid(True)
plt.show()
