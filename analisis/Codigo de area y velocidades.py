import os
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours
from Analisis_poco_contorno import ImageEnhancer

# =========================================
# PARTE 1 – Cargar imágenes y contornos
# =========================================

def cargar_imagenes_en_orden(directorio, patron=r'resta_(\d{8}_\d{6})\.tif'):
    regex = re.compile(patron)
    archivos = []
    for nombre in os.listdir(directorio):
        if regex.match(nombre):
            timestamp = regex.match(nombre).group(1)
            archivos.append((timestamp, nombre))
    archivos.sort()
    imagenes = [imread(os.path.join(directorio, nombre)) for _, nombre in archivos]
    return imagenes, [nombre for _, nombre in archivos]

def extraer_contornos(imagenes, **kwargs):
    contornos_por_frame = []
    for img in imagenes:
        enh = ImageEnhancer(img)
        _, contornos, _ = enh.procesar(mostrar=False, **kwargs)
        contornos_por_frame.append(contornos)
    return contornos_por_frame

# =========================================
# PARTE 2 – Calcular área y perímetro de cada contorno
# =========================================

def area_poligono(puntos):
    """Calcula el área de un polígono usando la fórmula del shoelace"""
    x = puntos[:, 1]
    y = puntos[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calcular_perimetro_contorno(contorno):
    """Calcula el perímetro de un solo contorno"""
    if len(contorno) < 2:
        return 0.0
    
    # Calcular las diferencias entre puntos consecutivos
    diffs = np.diff(contorno, axis=0, prepend=contorno[-1:])
    
    # Calcular la distancia euclidiana entre cada par de puntos consecutivos
    distancias = np.sqrt(np.sum(diffs**2, axis=1))
    
    return np.sum(distancias)

def calcular_perimetros_por_frame(contornos_por_frame):
    """Calcula los perímetros totales para cada frame"""
    perimetros = []
    for contornos in contornos_por_frame:
        perimetro_total = 0.0
        for c in contornos:
            if len(c) >= 2:  # Necesitamos al menos 2 puntos para un perímetro
                perimetro_total += calcular_perimetro_contorno(c)
        perimetros.append(perimetro_total)
    return np.array(perimetros)

def calcular_areas_por_frame(contornos_por_frame):
    """Calcula las áreas totales para cada frame"""
    areas = []
    for contornos in contornos_por_frame:
        area_total = 0.0
        for c in contornos:
            if len(c) >= 3:  # Necesitamos al menos 3 puntos para un área
                area_total += area_poligono(c)
        areas.append(area_total)
    return np.array(areas)

# =========================================
# PARTE 3 – Graficar resultados
# =========================================

def plot_metricas_vs_frame(areas, perimetros, nombres=None):
    """Grafica áreas y perímetros en función del frame"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Gráfico de áreas
    ax1.plot(areas, 'o-', color='blue', label="Área total")
    ax1.set_ylabel("Área total (pix²)")
    ax1.set_title("Evolución del área y perímetro del dominio")
    ax1.grid(True)
    ax1.legend()
    
    # Gráfico de perímetros
    ax2.plot(perimetros, 's-', color='red', label="Perímetro total")
    ax2.set_ylabel("Perímetro total (pix)")
    ax2.grid(True)
    ax2.legend()
    
    if nombres:
        ticks = range(len(nombres))
        ax1.set_xticks(ticks)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(nombres, rotation=45)
    else:
        ax2.set_xlabel("Frame")
    
    plt.tight_layout()
    plt.show()

# =========================================
# MAIN
# =========================================

# Ruta a las imágenes
directorio = r"D:\Labos 6-7 2025\Baut+Toto\23-06-2025\260 x 5 relocated"

# Recorte opcional
recorte = (slice(570, 710), slice(830, 970))  # [fila, columna]

# Cargar y recortar
imagenes, nombres = cargar_imagenes_en_orden(directorio)
imagenes = [img[recorte] for img in imagenes]
N = 12
imagenes = imagenes[N:]

# Detectar contornos
contornos_por_frame = extraer_contornos(
    imagenes,
    suavizado=3,
    percentil_contornos=99.9,
    min_dist_picos=8000,
    metodo_contorno="binarizacion"
)

# Calcular métricas
areas = calcular_areas_por_frame(contornos_por_frame)
perimetros = calcular_perimetros_por_frame(contornos_por_frame)

# Graficar resultados
plot_metricas_vs_frame(areas, perimetros, nombres)

#%%

desplazamiento = []

for i in range(len(areas)-1):
    A1 = areas[i]
    A2 = areas[i+1]
    p1 = perimetros[i]
    p2 = perimetros[i+1]
    p = (p1 + p2)/2
    
    dist = (A2 - A1)/p
    
    desplazamiento.append(dist)
    
desplazamiento = np.array(desplazamiento) / 5

plt.plot(desplazamiento)
plt.xlabel('numerito')
plt.ylabel('metrica 1 de area')


#%%

def mostrar_imagenes_con_contornos(imagenes, contornos_por_frame, nombres=None):
    for i, (img, contornos) in enumerate(zip(imagenes, contornos_por_frame)):
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        for c in contornos:
            plt.plot(c[:, 1], c[:, 0], linewidth=0.8, c='cyan')
        nombre = nombres[i] if nombres else f'Frame {i}'
        plt.title(f"{nombre} - {len(contornos)} contornos")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

mostrar_imagenes_con_contornos(imagenes, contornos_por_frame, nombres)

#%%

im_nueva = imagenes[27] - imagenes[20]

im_norm = (im_nueva - im_nueva.min())/(im_nueva.max() - im_nueva.min())

plt.imshow(im_norm, cmap='gray')
plt.colorbar()

#%%

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours
from scipy.spatial import cKDTree

# =========================================
# PARTE 1 – Cargar imágenes y contornos
# =========================================

def cargar_imagenes_en_orden(directorio, patron=r'resta_(\d{8}_\d{6})\.tif'):
    regex = re.compile(patron)
    archivos = []
    for nombre in os.listdir(directorio):
        if regex.match(nombre):
            timestamp = regex.match(nombre).group(1)
            archivos.append((timestamp, nombre))
    archivos.sort()
    imagenes = [imread(os.path.join(directorio, nombre)) for _, nombre in archivos]
    return imagenes, [nombre for _, nombre in archivos]

def extraer_contornos(imagenes, **kwargs):
    contornos_por_frame = []
    for img in imagenes:
        enh = ImageEnhancer(img)
        _, contornos, _ = enh.procesar(mostrar=False, **kwargs)
        contornos_por_frame.append(contornos)
    return contornos_por_frame

# =========================================
# PARTE 2 – Trackeo acumulado de puntos
# =========================================

def calcular_desplazamiento_total(contornos_por_imagen, distancia_maxima=15.0):
    puntos_iniciales = np.vstack(contornos_por_imagen[0])
    puntos_actuales = puntos_iniciales.copy()

    for i in range(1, len(contornos_por_imagen)):
        puntos_siguientes = np.vstack(contornos_por_imagen[i])
        tree = cKDTree(puntos_siguientes)
        distancias, indices = tree.query(puntos_actuales, distance_upper_bound=distancia_maxima)
        validos = distancias != np.inf
        nuevos_puntos = puntos_actuales.copy()
        nuevos_puntos[validos] = puntos_siguientes[indices[validos]]
        puntos_actuales = nuevos_puntos

    desplazamientos = puntos_actuales - puntos_iniciales
    return puntos_iniciales, desplazamientos

# =========================================
# PARTE 3 – Cálculo de velocidad radial
# =========================================

def calcular_velocidad_radial(puntos_iniciales, desplazamientos, centro):
    radios = puntos_iniciales - centro
    radios_unit = radios / np.linalg.norm(radios, axis=1)[:, np.newaxis]
    v_radial = np.sum(desplazamientos * radios_unit, axis=1)
    radio_inicial = np.linalg.norm(radios, axis=1)
    return radio_inicial, v_radial

def plot_velocidad_radial_vs_radio(radio_inicial, v_radial):
    plt.figure(figsize=(8, 5))
    plt.scatter(radio_inicial, v_radial, c=v_radial, cmap='plasma', alpha=0.6)
    plt.xlabel("Distancia radial al centro (pix)")
    plt.ylabel("Velocidad radial (pix / total_frames)")
    plt.title("Velocidad radial en función del radio")
    plt.colorbar(label="Velocidad radial")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =========================================
# MAIN
# =========================================

# Ruta a tus imágenes:
directorio = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\23-06-2025\250 x 10"

# Centro fijo para calcular la velocidad radial
centro = np.array([84, 82])  # coordenadas (fila, columna)

# Recorte de las imágenes (ajustalo si querés)
recorte = (slice(100, 285), slice(40, 225))

# Cargar y recortar imágenes
imagenes, nombres = cargar_imagenes_en_orden(directorio)
imagenes = [img[recorte] for img in imagenes]

# Extraer contornos
contornos_por_frame = extraer_contornos(
    imagenes,
    suavizado=5,
    percentil_contornos=99.99,
    min_dist_picos=300,
    metodo_contorno="binarizacion"
)

# Calcular desplazamientos acumulados
p_ini, desplazamientos = calcular_desplazamiento_total(contornos_por_frame, distancia_maxima=10)

# Calcular velocidades radiales
radio_inicial, v_radial = calcular_velocidad_radial(p_ini, desplazamientos, centro)

# Graficar resultado
plot_velocidad_radial_vs_radio(radio_inicial, v_radial)


