import os, re
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours, perimeter
from skimage.filters import threshold_otsu, gaussian
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter

# ------------------------------------------------------------
# Clase para mejorar imágenes no binarizadas
# ------------------------------------------------------------
class ImageEnhancer:
    def __init__(self, imagen, sigma_background=100, alpha=0):
        self.image = imagen
        self.sigma_background = sigma_background
        self.alpha = alpha

    def _subtract_background(self):
        background = gaussian(self.image.astype(np.float32),
                              sigma=self.sigma_background,
                              preserve_range=True)
        corrected = self.image.astype(np.float32) - self.alpha * background
        return corrected

    def _find_large_contours(self, binary, percentil_contornos=99.9):
        contours = find_contours(binary, level=0.5)
        if percentil_contornos > 0 and contours:
            def area_contour(contour):
                x = contour[:, 1]
                y = contour[:, 0]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas = np.array([area_contour(c) for c in contours])
            umbral = np.percentile(areas, percentil_contornos)
            return [c for c, a in zip(contours, areas) if a >= umbral]
        return contours

    def procesar(self, suavizado=3, percentil_contornos=99.9):
        corrected = self._subtract_background() if self.alpha > 0 else self.image.astype(float)
        smooth = uniform_filter(corrected, size=suavizado)
        threshold = threshold_otsu(smooth)
        binary = (smooth > threshold).astype(np.uint16)
        contours = self._find_large_contours(binary, percentil_contornos=percentil_contornos)
        return binary, contours

# ------------------------------------------------------------
# Cálculo de Var(u)
# ------------------------------------------------------------
def var_u(binary1, binary2, contours1):
    delta_a = binary2.astype(int) - binary1.astype(int)
    changed_pixels = np.argwhere(delta_a != 0)

    if len(changed_pixels) == 0 or len(contours1) == 0:
        return 0.0

    cont1 = np.vstack(contours1)
    P = perimeter(binary1)
    if P == 0:
        return 0.0

    tree = cKDTree(cont1)
    distances, _ = tree.query(changed_pixels)

    sum_uprime = np.sum(distances)
    sum_abs_da = np.sum(np.abs(delta_a))

    var_u_value = (2 * sum_uprime / P) - (sum_abs_da / P) ** 2
    return var_u_value

# ------------------------------------------------------------
# Cargar imágenes desde carpeta
# ------------------------------------------------------------
def load_images_folder(folder, regex_pattern, min_index=15):
    pattern = re.compile(regex_pattern)
    files = []

    for f in os.listdir(folder):
        match = pattern.match(f)
        if match:
            try:
                key = int(match.group(1))
            except:
                key = match.group(1)
            files.append((key, f))

    files.sort(key=lambda x: x[0])
    files = files[min_index:]

    images, filenames = [], []
    for _, fname in files:
        try:
            img = imread(os.path.join(folder, fname))
            images.append(img)
            filenames.append(fname)
        except:
            print(f"No se pudo cargar {fname}")
    return images, filenames

# ------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------
def pipeline(folder, regex_pattern, already_binary=False):
    images, filenames = load_images_folder(folder, regex_pattern, min_index=15)

    variances = []
    for k in range(len(images) - 1):
        if already_binary:
            binary1 = (images[k] > 0).astype(np.uint8)
            binary2 = (images[k+1] > 0).astype(np.uint8)
            contours1 = find_contours(binary1, level=0.5)
        else:
            enhancer1 = ImageEnhancer(images[k], sigma_background=100, alpha=0)
            enhancer2 = ImageEnhancer(images[k+1], sigma_background=100, alpha=0)
            binary1, contours1 = enhancer1.procesar(suavizado=3, percentil_contornos=99.9)
            binary2, _ = enhancer2.procesar(suavizado=3, percentil_contornos=99.9)

        v = var_u(binary1, binary2, contours1)
        variances.append(v)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(variances)), variances, marker="o")
    plt.xlabel("Paso")
    plt.ylabel("Var(u)")
    plt.title(f"Varianza del desplazamiento en {os.path.basename(folder)}")
    plt.grid(True)
    plt.show()

    return variances

# ------------------------------------------------------------
# Ejecución en carpetas
# ------------------------------------------------------------
folder1 = r"C:\Users\Marina\Documents\Labo 6\imagenes\250 x 10"
regex1 = r'resta_(\d{8}_\d{6})\.tif'
var1 = pipeline(folder1, regex1, already_binary=False)

folder2 = r"C:\Users\Marina\Documents\Labo 6\imagenes\test_velocidades"
regex2 = r".*-(\d+).tif$"
var2 = pipeline(folder2, regex2, already_binary=True)