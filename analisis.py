import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import gaussian, sobel, threshold_otsu, median
from skimage.morphology import disk
from skimage.restoration import rolling_ball
from skimage.morphology import white_tophat
from scipy.ndimage import uniform_filter
from IPython import get_ipython

# Activar modo gráfico externo
get_ipython().run_line_magic('matplotlib', 'qt5')

# Cargar imagen
carpeta = r"C:\Users\Marina\Documents\Labo 6\LuCam-app\test\test fotos"
nombre_imagen = "Resta-P8139-150Oe-50ms-89.tif"
ruta_imagen = os.path.join(carpeta, nombre_imagen)
image = imread(ruta_imagen)

#---------------------- Procesamientos ----------------------

# Filtros para reducción de ruido
gaussian_noise = gaussian(image, sigma=2, preserve_range=True)
median_noise = median(image, footprint=disk(4))

# Remoción de fondo
background_rolling = rolling_ball(image, radius=50)
image_rolling = image - background_rolling

background_tophat = white_tophat(image, footprint=disk(50))

background_gaussian = gaussian(image, sigma=50, preserve_range=True)
image_background_gauss = image - background_gaussian

# Aumento de contraste
norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))
contrast_tanh = np.tanh(4 * (norm_image - 0.5))
contrast_tanh = ((contrast_tanh + 1) / 2 * 255).astype(np.uint8)

contrast_sigmoid = 2 / (1 + np.exp(-4 * (norm_image - 0.5))) - 1
contrast_sigmoid = ((contrast_sigmoid + 1) / 2 * 255).astype(np.uint8)

# Binarización
thresh_original = threshold_otsu(image)
binary_original = (image > thresh_original).astype(np.uint8) * 255

thresh_sigmoid = threshold_otsu(contrast_sigmoid)
binary_sigmoid = (contrast_sigmoid > thresh_sigmoid).astype(np.uint8) * 255

#---------------------- Gráfico Final ----------------------

fig = plt.figure(figsize=(30, 18))  # <<< MÁS GRANDE
grid = plt.GridSpec(6, 6, wspace=0.4, hspace=0.4)

# Imagen original grande (ocupa toda la primera columna)
ax_orig = fig.add_subplot(grid[:, 0])
ax_orig.imshow(image, cmap='gray')
ax_orig.set_title("Imagen Original", fontsize=20)
ax_orig.axis('off')

# Noise Reduction
ax1 = fig.add_subplot(grid[0, 1])
ax1.imshow(gaussian_noise, cmap='gray')
ax1.set_title("Gaussiano (σ=2)", fontsize=16)
ax1.axis('off')

ax2 = fig.add_subplot(grid[1, 1])
ax2.imshow(median_noise, cmap='gray')
ax2.set_title("Mediana (r=4)", fontsize=16)
ax2.axis('off')

# Background Removal
ax3 = fig.add_subplot(grid[0, 2])
ax3.imshow(image_rolling, cmap='gray')
ax3.set_title("Rolling Ball", fontsize=16)
ax3.axis('off')

ax4 = fig.add_subplot(grid[1, 2])
ax4.imshow(background_tophat, cmap='gray')
ax4.set_title("Top-Hat", fontsize=16)
ax4.axis('off')

ax5 = fig.add_subplot(grid[2, 2])
ax5.imshow(image_background_gauss, cmap='gray')
ax5.set_title("Fondo Gaussiano", fontsize=16)
ax5.axis('off')

# Contrast Enhancement
ax6 = fig.add_subplot(grid[0, 3])
ax6.imshow(contrast_tanh, cmap='gray')
ax6.set_title("Contraste Tanh", fontsize=16)
ax6.axis('off')

ax7 = fig.add_subplot(grid[1, 3])
ax7.imshow(contrast_sigmoid, cmap='gray')
ax7.set_title("Contraste Sigmoide", fontsize=16)
ax7.axis('off')

# Binarización
ax8 = fig.add_subplot(grid[0, 4])
ax8.imshow(binary_original, cmap='gray')
ax8.set_title(f"Binarización Original\n(Otsu={thresh_original:.2f})", fontsize=16)
ax8.axis('off')

ax9 = fig.add_subplot(grid[1, 4])
ax9.imshow(binary_sigmoid, cmap='gray')
ax9.set_title(f"Binarización Sigmoide\n(Otsu={thresh_sigmoid:.2f})", fontsize=16)
ax9.axis('off')

plt.show()
