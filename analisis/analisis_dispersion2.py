# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 11:42:07 2025

@author: Toto y Bauti
"""

import os, re
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours, perimeter
from skimage.filters import threshold_otsu, gaussian
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter

# ------------------------------------------------------------
# Preprocesamiento y binarización
# ------------------------------------------------------------
def preprocess_image(image, already_binary=False, smooth=3):
    """
    Convierte la imagen a binaria.
    - Si ya es binaria: asegura que esté en 0/1.
    - Si no: aplica suavizado y threshold de Otsu.
    """
    img = image.astype(np.float32)

    if already_binary:
        # Aseguramos que sea 0/1
        binary = (img > 0).astype(np.uint8)
    else:
        # Suavizado y umbral Otsu
        smooth_img = uniform_filter(img, size=smooth)
        thresh = threshold_otsu(smooth_img)
        binary = (smooth_img > thresh).astype(np.uint8)

    # Contornos del dominio
    contours = find_contours(binary, level=0.5)
    return binary, contours

# ------------------------------------------------------------
# Cálculo de Var(u)
# ------------------------------------------------------------
def var_u(binary1, binary2, contours1):
    """
    Calcula la varianza del desplazamiento Var(u) entre dos imágenes binarias.
    """
    delta_a = binary2.astype(int) - binary1.astype(int)   # cambios de +1/-1/0
    changed_pixels = np.argwhere(delta_a != 0)

    if len(changed_pixels) == 0 or len(contours1) == 0:
        return 0.0

    # Concatenamos todos los contornos de la primera imagen
    cont1 = np.vstack(contours1)
    P = perimeter(binary1)
    if P == 0:
        return 0.0

    # Distancias mínimas de píxeles cambiados al contorno original
    tree = cKDTree(cont1)
    distances, _ = tree.query(changed_pixels)

    # Aplicamos la fórmula
    sum_uprime = np.sum(distances)
    sum_abs_da = np.sum(np.abs(delta_a))  # solo |Δa_i|

    var_u_value = (2 * sum_uprime / P) - (sum_abs_da / P) ** 2
    return var_u_value

# ------------------------------------------------------------
# Cargar imágenes de una carpeta
# ------------------------------------------------------------
def load_images_folder(folder, regex_pattern):
    """
    Carga imágenes de una carpeta y las ordena usando un número en el nombre.
    """
    pattern = re.compile(regex_pattern)
    files = []

    for f in os.listdir(folder):
        match = pattern.match(f)
        if match:
            # key es el número que define el orden
            try:
                key = int(match.group(1))
            except:
                key = match.group(1)
            files.append((key, f))

    files.sort(key=lambda x: x[0])

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
    """
    Ejecuta todo el cálculo de Var(u) para una carpeta.
    """
    images, filenames = load_images_folder(folder, regex_pattern)

    binaries, contours = [], []
    for img in images:
        binary, conts = preprocess_image(img, already_binary=already_binary, smooth=3)
        binaries.append(binary)
        contours.append(conts)

    # Calcular Var(u) entre frames consecutivos
    variances = []
    for k in range(len(binaries) - 1):
        v = var_u(binaries[k], binaries[k+1], contours[k])
        variances.append(v)

    # Graficar resultados
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(variances)), variances, marker="o")
    plt.xlabel("Paso")
    plt.ylabel("Var(u)")
    plt.title(f"Varianza del desplazamiento en {os.path.basename(folder)}")
    plt.grid(True)
    plt.show()

    return variances

# ------------------------------------------------------------
# Uso en las dos carpetas
# ------------------------------------------------------------
# Caso 1: imágenes NO binarizadas (usar Otsu)
folder1 = r"C:\Users\Marina\Documents\Labo 6\imagenes\250 x 10"
regex1 = r'resta_(\d{8}_\d{6})\.tif'
var1 = pipeline(folder1, regex1, already_binary=False)

# Caso 2: imágenes YA binarizadas
folder2 = r"C:\Users\Marina\Documents\Labo 6\imagenes\test_velocidades"
regex2 = r".*-(\d+).tif$"
var2 = pipeline(folder2, regex2, already_binary=True)
