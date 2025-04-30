#%%

from skimage.io import imread
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import find_contours, regionprops, label
from scipy.ndimage import uniform_filter
import numpy as np
import matplotlib.pyplot as plt

# Parámetros ajustables
ruta_imagen = "C:/Users/Tomas/Desktop/FACULTAD/LABO 6/Resta-P8139-150Oe-50ms-1000.tif"
sigma_background = 200   # Suavizado para estimar fondo
alpha = 1                # Peso de la sustracción de fondo
ganancia_gaussiana = 60    # Ganancia más chica
ganancia_tanh = 0.008     # Ganancia más grande
ganancia_lorentz = 50
suavizado = 5     # Tamaño del filtro uniform_filter

# ------------------ Procesamiento ------------------

# Cargar imagen
image = imread(ruta_imagen)

# Estimar fondo
background = gaussian(image.astype(np.float32), sigma=sigma_background, preserve_range=True)

# Sustracción parcial del fondo
corrected = image.astype(np.float32) - alpha * background
corrected = np.clip(corrected, 0, None)


# Realce con dos tangentes hiperbólicas
delta = corrected - 255/2
enhanced = np.exp(-(delta/ganancia_gaussiana)**2)
#enhanced = ganancia_lorentz/(np.pi*(delta**2 + ganancia_lorentz**2))


# Normalizar a [0, 1]
enhanced_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

# Reescalar a 8 bits
enhanced_uint8 = (enhanced_norm * 255).astype(np.uint8)

smooth = uniform_filter(enhanced_uint8,size=suavizado)

delta2 = smooth - np.mean(smooth)
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

plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
for contours_filtrados in contours_filtrados:
    plt.plot(contours_filtrados[:, 1], contours_filtrados[:, 0], linewidth=1, color='cyan')
plt.title("Original + contornos")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(corrected, cmap='gray')
plt.title("Con fondo restado")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(smooth, cmap='gray')
plt.title("Suavizada con media movil")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(enhanced_uint8, cmap='gray')
plt.title("Gaussiana")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(enhanced2_uint8, cmap='gray')
plt.title("Tanh")
plt.axis('off')


plt.subplot(2, 3, 6)
plt.imshow(binary, cmap='gray')
for contorno in contours_filtrados:
    if contorno.ndim == 2 and contorno.shape[1] >= 2:
        plt.plot(contorno[:, 1], contorno[:, 0], linewidth=1, color='cyan')
plt.title(f"Contornos ({len(contours_filtrados)} detectados)")
plt.axis('off')

plt.tight_layout()
plt.show()

print(np.mean(smooth))

#%%

image1 = imread(ruta_imagen)
image = image1[400:700,475:825]

# Estimar fondo
background = gaussian(image.astype(np.float32), sigma=sigma_background, preserve_range=True)

# Sustracción parcial del fondo
corrected = image.astype(np.float32) - alpha * background
corrected = np.clip(corrected, 0, None)


# Realce con dos tangentes hiperbólicas
delta = corrected - 255/2 + np.mean(corrected)
ganancia_gaussiana = 2*np.std(corrected)
enhanced = np.exp(-(delta/ganancia_gaussiana)**2)
#enhanced = ganancia_lorentz/(np.pi*(delta**2 + ganancia_lorentz**2))


# Normalizar a [0, 1]
enhanced_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

# Reescalar a 8 bits
enhanced_uint8 = (enhanced_norm * 255).astype(np.uint8)

suavizado = 2
smooth = uniform_filter(enhanced_uint8,size=suavizado)

delta2 = smooth - np.mean(smooth)
ganancia_tanh = 0.1
enhanced2 = 0.5 * (np.tanh(ganancia_tanh * delta2) + 1)


# Normalizar a [0, 1]
enhanced2_norm = (enhanced2 - enhanced2.min()) / (enhanced2.max() - enhanced2.min())

# Reescalar a 8 bits
enhanced2_uint8 = (enhanced2_norm * 255).astype(np.uint8)

# Umbral automático con Otsu
threshold = threshold_otsu(enhanced2_uint8)

# Binarizar
binary = (enhanced2_uint8 > threshold).astype(np.uint8) * 255



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
percentil = 99.9
area_umbral = np.percentile(areas_contornos, percentil)

# Filtrar contornos grandes
contours_filtrados = [c for c, area in zip(contours, areas_contornos) if area >= area_umbral]

# ------------------ Mostrar resultados ------------------

plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
for contorno in contours_filtrados:
    plt.plot(contorno[:, 1], contorno[:, 0], linewidth=1, color='cyan')
plt.title("Original + contornos")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(corrected, cmap='gray')
plt.title("Con fondo restado")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(smooth, cmap='gray')
plt.title("Suavizada con media movil")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(enhanced_uint8, cmap='gray')
plt.title("Gaussiana")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(enhanced2_uint8, cmap='gray')
plt.title("Tanh")
plt.axis('off')


plt.subplot(2, 3, 6)
plt.imshow(binary, cmap='gray')
for contorno in contours_filtrados:
    plt.plot(contorno[:, 1], contorno[:, 0], linewidth=1, color='cyan')
plt.title(f"Contornos ({len(contours_filtrados)} detectados)")
plt.axis('off')

plt.tight_layout()
plt.show()

print(ganancia_gaussiana)
print(np.mean(corrected))

#%%

from skimage.io import imread
from skimage.filters import sobel, roberts
from skimage.feature import canny
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import numpy as np

# Parámetro de percentil
percentil = 0

im = gaussian(binary,sigma=3)

# Aplicar filtros de detección de bordes
edges_canny = canny(im, sigma=1)
edges_sobel = sobel(im)
edges_roberts = roberts(im)

# Obtener contornos de cada detector
contornos_canny = find_contours(edges_canny, level=0.1)
contornos_sobel = find_contours(edges_sobel, level=0.1)
contornos_roberts = find_contours(edges_roberts, level=0.1)


# Mostrar resultados
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
for c in contornos_canny:
    plt.plot(c[:, 1], c[:, 0], color='red', linewidth=1)
plt.title(f"Canny - Contornos > {percentil}° pct")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image, cmap='gray')
for c in contornos_sobel:
    plt.plot(c[:, 1], c[:, 0], color='blue', linewidth=1)
plt.title(f"Sobel - Contornos > {percentil}° pct")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image, cmap='gray')
for c in contornos_roberts:
    plt.plot(c[:, 1], c[:, 0], color='green', linewidth=1)
plt.title(f"Roberts - Contornos > {percentil}° pct")
plt.axis('off')

plt.tight_layout()
plt.show()



#%%
import numpy as np
import matplotlib.pyplot as plt

def plot_gaussian_and_tanh(ganancia_gaussiana, ganancia_tanh, ganancia_lorentz, ganancia,  x_min=0, x_max=255, puntos=1000):
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
    delta = x - (x_max / 2)  + 43
    delta1 = x - 36
    
    # Calcular funciones
    y_gauss = np.exp(-(delta/ganancia_gaussiana)**2)
    y_tanh = 0.5 * (np.tanh(ganancia_tanh * delta1) + 1)
    y_lorentz = ganancia_lorentz/(np.pi*(delta**2 + ganancia_lorentz**2))
    y_negro = delta1/(np.exp(delta1*ganancia)-1)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_gauss, label=f'Gaussiana invertida (ganancia={ganancia_gaussiana})', color='blue')
    plt.plot(x, y_tanh, label=f'Tanh (ganancia={ganancia_tanh})', color='red')
    plt.plot(x, y_lorentz, label=f'Lorentz (ganancia={ganancia_lorentz})', color='green')
    #plt.plot(x, y_negro, label=f'Negro (ganancia={ganancia})', color='orange')
    plt.title('Funciones: Gaussiana Invertida y Tanh')
    plt.xlabel('Intensidad (x)')
    plt.ylabel('Valor de la función')
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejemplo de uso:
plot_gaussian_and_tanh(ganancia_gaussiana=42.871 , ganancia_tanh=0.1, ganancia_lorentz = 100, ganancia = 1000000)