# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:04:32 2025

@author: Marina
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

saturado_positivo = imread("D:/Labos 6-7 2025/Baut+Toto/Fotos Test/Fotos 21-5/Saturado positivo.tif")
saturado_negativo = imread("D:/Labos 6-7 2025/Baut+Toto/Fotos Test/Fotos 21-5/Saturado negativo.tif")
raw = imread("D:/Labos 6-7 2025/Baut+Toto/Fotos Test/raw/cruda_20250521_125452.tif")

def remove_horizontal_lines_local(img):
    row_means = np.mean(img, axis=1, keepdims=True)
    corrected = img - row_means
    corrected += np.mean(img)  # conservar brillo promedio
    return corrected

raw = remove_horizontal_lines_local(raw)

resta = saturado_positivo - saturado_negativo

prueba = (raw - remove_horizontal_lines_local(saturado_negativo))/2*resta


prueba_norm = (prueba - prueba.min()) / (prueba.max() - prueba.min())
prueba_uint16 = (prueba_norm * 65535).astype(np.uint16)


plt.figure(figsize=(12, 4))
plt.imshow(prueba_uint16, cmap='gray')
plt.title("Resta")
plt.axis('off')
