#%%

from skimage.io import imread
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import find_contours, regionprops, label
from scipy.ndimage import uniform_filter
import numpy as np
import matplotlib.pyplot as plt

# Parámetros ajustables
ruta_imagen = "D:/Labos 6-7 2025/Baut+Toto/Resta-P8139-150Oe-50ms-88.tif"
sigma_background = 200   # Suavizado para estimar fondo
alpha = 1                # Peso de la sustracción de fondo
ganancia_gauss = -0.0001    # Ganancia más chica
ganancia_tanh = 0.05     # Ganancia más grande
suavizado_pre = 3     # Tamaño del filtro uniform_filter

# ------------------ Procesamiento ------------------

# Cargar imagen
image = imread(ruta_imagen)

# Estimar fondo
background = gaussian(image.astype(np.float32), sigma=sigma_background, preserve_range=True)

# Sustracción parcial del fondo
corrected = image.astype(np.float32) - alpha * background
corrected = np.clip(corrected, 0, None)

corrected_smooth = gaussian(corrected, sigma=suavizado_pre)

# Realce con dos tangentes hiperbólicas
delta = corrected_smooth - 255/2
enhanced = np.exp(ganancia_gauss*(delta)**2)


# Normalizar a [0, 1]
enhanced_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

# Reescalar a 8 bits
enhanced_uint8 = (enhanced_norm * 255).astype(np.uint8)


delta2 = enhanced_uint8 - 255/2
enhanced2 = 0.5 * (np.tanh(ganancia_tanh * delta2) + 1)


# Normalizar a [0, 1]
enhanced2_norm = (enhanced2 - enhanced2.min()) / (enhanced2.max() - enhanced2.min())

# Reescalar a 8 bits
enhanced2_uint8 = (enhanced2_norm * 255).astype(np.uint8)


# Umbral automático con Otsu
threshold = threshold_otsu(enhanced2_uint8)

# Binarizar
binary = (enhanced2_uint8 > threshold).astype(np.uint8) * 255

from skimage.morphology import opening, disk, closing

binary = opening(binary, disk(3))
binary = closing(binary, disk(3))

# ------------------ Encontrar y filtrar contornos ------------------

# Encontrar contornos
contours = find_contours(binary, level=0.5)

def area_contorno(contour):
    x = contour[:, 1]
    y = contour[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Calcular áreas de los contornos
areas_contornos = np.array([area_contorno(c) for c in contours])

# Definir percentil mínimo
percentil = 90
area_umbral = np.percentile(areas_contornos, percentil)

# Filtrar contornos grandes
contours_filtrados = [c for c, area in zip(contours, areas_contornos) if area >= area_umbral]

# ------------------ Mostrar resultados ------------------

plt.figure(figsize=(20, 5))

plt.subplot(1, 6, 1)
plt.imshow(image, cmap='gray')
for contours_filtrados in contours_filtrados:
    plt.plot(contours_filtrados[:, 1], contours_filtrados[:, 0], linewidth=1, color='cyan')
plt.title(f"Original + contornos")
plt.axis('off')

plt.subplot(1, 6, 2)
plt.imshow(corrected_smooth, cmap='gray')
plt.title("Fondo restado y filtrado")
plt.axis('off')

plt.subplot(1, 6, 3)
plt.imshow(enhanced_uint8, cmap='gray')
plt.title(f"Gaussiana")
plt.axis('off')

plt.subplot(1, 6, 4)
plt.imshow(enhanced2_uint8, cmap='gray')
plt.title(f"Tanh")
plt.axis('off')


plt.subplot(1, 6, 5)
plt.imshow(binary, cmap='gray')
for contours_filtrados in contours_filtrados:
    plt.plot(contours_filtrados[:, 1], contours_filtrados[:, 0], linewidth=1, color='cyan')
plt.title(f"Contornos ({len(contours)} detectados)")
plt.axis('off')

plt.tight_layout()
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

def plot_gaussian_and_tanh(ganancia_gaussiana, ganancia_tanh, x_min=0, x_max=255, puntos=1000):
    """
    Plotea:
    - Una gaussiana invertida (exp(-ganancia_gaussiana*(x-127.5)^2))
    - Una tangente hiperbólica centrada en 127.5
    
    Parámetros:
    - ganancia_gaussiana: float (negativa para que decaiga)
    - ganancia_tanh: float
    - x_min, x_max: rango del eje x
    - puntos: cantidad de puntos
    """
    # Crear eje x
    x = np.linspace(x_min, x_max, puntos)
    
    # Delta centrado en la mitad (127.5)
    delta = x - (x_max / 2)
    
    # Calcular funciones
    y_gauss = np.exp(ganancia_gaussiana * (delta)**2)
    y_tanh = 0.5 * (np.tanh(ganancia_tanh * delta) + 1)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_gauss, label=f'Gaussiana invertida (ganancia={ganancia_gaussiana})', color='blue')
    plt.plot(x, y_tanh, label=f'Tanh (ganancia={ganancia_tanh})', color='red')
    plt.title('Funciones: Gaussiana Invertida y Tanh')
    plt.xlabel('Intensidad (x)')
    plt.ylabel('Valor de la función')
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejemplo de uso:
plot_gaussian_and_tanh(ganancia_gaussiana=-0.0005, ganancia_tanh=0.03)
