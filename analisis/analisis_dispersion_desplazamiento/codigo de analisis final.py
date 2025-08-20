import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours
from scipy.stats import linregress

# --------------------------
# Parámetros configurables
# --------------------------
RAIZ = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe"  # <-- Cambiá esta ruta
CAMPOS = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]  # en Oe
TIEMPOS = {
    80: 100,
    90: 50,
    100: 30,
    110: 20,
    120: 10,
    130: 10,
    140: 5,
    150: 3,
    170: 2,
    180: 1,
    190: 1,
    200: 0.5,
}  # tiempo entre frames en ms
ESCALA = 0.3 / 1000000  # micrones por pixel

# --------------------------
def calcular_area(contour):
    x, y = contour[:, 1], contour[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calcular_perimetro(contour):
    diffs = np.diff(contour, axis=0, prepend=contour[-1:])
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

def estimar_radio_suave(binary_image, centro):
    contours = find_contours(binary_image.astype(float), level=0.5)
    if not contours:
        return 0
    contorno_principal = max(contours, key=len)
    dx = contorno_principal[:, 1] - centro[0]
    dy = contorno_principal[:, 0] - centro[1]
    radios = np.sqrt(dx**2 + dy**2)
    return np.mean(radios)

def analizar_carpeta(campo, centro):
    carpeta = os.path.join(RAIZ, f"{campo:03d}")
    tiempo_ms = TIEMPOS[campo]
    areas, perimetros, radios = [], [], []

    for i in range(6):  # 0 a 5
        patron = f"Bin-P8137-{campo:03d}Oe-{tiempo_ms}ms-{i}.tif"
        path = os.path.join(carpeta, patron)
        if not os.path.exists(path):
            print(f"Falta: {path}")
            continue
        imagen = imread(path)
        contornos = find_contours(imagen.astype(float), level=0.5)

        A = sum(calcular_area(c) for c in contornos if len(c) >= 3)
        P = sum(calcular_perimetro(c) for c in contornos if len(c) >= 3)
        R_suave = estimar_radio_suave(imagen, centro)

        areas.append(A)
        perimetros.append(P)
        radios.append(R_suave)

    areas = np.array(areas)
    perimetros = np.array(perimetros)
    radios = np.array(radios)
    dt = tiempo_ms / 1000  # a segundos

    # Método 1: tu método
    delta_A = areas[1:] - areas[:-1]
    P_avg = (perimetros[1:] + perimetros[:-1]) / 2
    v1 = (delta_A / P_avg) / dt * ESCALA  # um/s

    # Método 2: circunferencia con radio suavizado
    delta_R = radios[1:] - radios[:-1]
    v2 = delta_R / dt * ESCALA  # um/s

    return campo, v1, v2

# --------------------------
resultados1 = []
resultados2 = []
errores1 = []
errores2 = []
H_inv_1_4 = []

# DEFINÍ EL CENTRO MANUALMENTE (en coordenadas x, y)
CENTRO = (153, 160)

for H in CAMPOS:
    try:
        campo, v1, v2 = analizar_carpeta(H, CENTRO)
        ln_v1 = np.log(v1)
        ln_v2 = np.log(v2)
        resultados1.append(np.mean(ln_v1))
        errores1.append(np.std(ln_v1) / np.sqrt(len(ln_v1)))
        resultados2.append(np.mean(ln_v2))
        errores2.append(np.std(ln_v2) / np.sqrt(len(ln_v2)))
        H_inv_1_4.append(H**(-0.25))
    except Exception as e:
        print(f"Error en H = {H}: {e}")

H_inv_1_4 = np.array(H_inv_1_4)
resultados1 = np.array(resultados1)
resultados2 = np.array(resultados2)
errores1 = np.array(errores1)
errores2 = np.array(errores2)

# --------------------------
# Ajuste lineal y correlación
# --------------------------
def ajustar_lineal(x, y):
    regresion = linregress(x, y)
    print(f"Pendiente: {regresion.slope:.4f} ± {regresion.stderr:.4f}")
    print(f"Intercepto: {regresion.intercept:.4f} ± {regresion.intercept_stderr:.4f}")
    print(f"R^2: {regresion.rvalue**2:.4f}")
    print(f"p-valor: {regresion.pvalue:.4e}\n")
    return regresion

print("Ajuste Método Área / Perímetro:")
ajuste1 = ajustar_lineal(H_inv_1_4, resultados1)

print("Ajuste Método Radio efectivo:")
ajuste2 = ajustar_lineal(H_inv_1_4, resultados2)

# --------------------------
# Graficar resultados
# --------------------------
#%%

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Estilo general
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": False
})

plt.figure(figsize=(6.2, 4.5))  # tamaño ideal para texto en LaTeX

# Gráfico con barras de error
plt.errorbar(H_inv_1_4, resultados1, yerr=errores1, fmt='o', 
             label='Área / Perímetro', capsize=3, markersize=5, color='navy')
plt.errorbar(H_inv_1_4, resultados2, yerr=errores2, fmt='s', 
             label='Radio efectivo', capsize=3, markersize=5, color='darkorange')

# Ajustes lineales
x_fit = np.linspace(H_inv_1_4.min(), H_inv_1_4.max(), 200)
plt.plot(x_fit, ajuste1.slope * x_fit + ajuste1.intercept, '--', color='navy', linewidth=1)
plt.plot(x_fit, ajuste2.slope * x_fit + ajuste2.intercept, '--', color='darkorange', linewidth=1)

# Etiquetas
plt.xlabel(r'$H^{-1/4}$ [$\mathrm{Oe}^{-1/4}$]')
plt.ylabel(r'$\ln(v)$ [ln(m/s)]')

plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend(frameon=False)
plt.tight_layout()

# Guardar para incluir en LaTeX
plt.savefig('ln_vel_vs_Hinv14.pdf')
plt.show()


#%%


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from Clase_de_areas_y_velocidades import ContourAnalysis

# ========== CONFIGURACIÓN ==========
RAIZ = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe\23-06-2025"  # <-- Cambiar
CAMPOS = {
    80: {
        "folder": "250 x 10",
        "delta_t": 0.010,
        "start": 15,
        "end": None,
        "crop": (slice(140, 280), slice(345, 495))
    },
    90: {
        "folder": "260 x 5",
        "delta_t": 0.005,
        "start": 9,
        "end": None,
        "crop": (slice(75, 275), slice(305, 539))
    },
    100: {
        "folder": "270 x 1",
        "delta_t": 0.001,
        "start": 6,
        "end": None,
        "crop": (slice(160, 210), slice(100, 160))
    },
    110: {
        "folder": "280 x 1",
        "delta_t": 0.001,
        "start": 6,
        "end": -4,
        "crop": (slice(875, 950), slice(785, 870))
    }
}
ESCALA = 0.36 / 1e6  # um/pixel
PATRON = r'resta_(\d{8}_\d{6})\.tif'

# ========== ANÁLISIS ==========
resultados_ln_v = []
errores_ln_v = []
H_inv_1_4 = []

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
            'percentil_contornos': 99.9,
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

    velocidades = []
    ln_vel = []
    varianzas = []

    delta_t = info['delta_t']

    for i in range(len(filenames) - 1):
        r1 = results[filenames[i]]
        r2 = results[filenames[i+1]]
        
        A1, A2 = r1['area'], r2['area']
        P1, P2 = r1['perimeter'], r2['perimeter']
        P_avg = (P1 + P2) / 2
        
        if P_avg > 0:
            desplazamiento = (A2 - A1) / P_avg
            v = np.abs(desplazamiento) / delta_t * ESCALA  # um/s
            velocidades.append(v)

    velocidades = np.array(velocidades)
    ln_v = np.log(velocidades)
    resultados_ln_v.append(np.mean(ln_v))
    errores_ln_v.append(np.std(ln_v) / np.sqrt(len(ln_v)))
    H_inv_1_4.append(campo ** (-0.25))

#%%


# ========== AJUSTE Y GRAFICO ==========
H_inv_1_4 = np.array(H_inv_1_4)
lnv = np.array(resultados_ln_v)
err = np.array(errores_ln_v)

regresion = linregress(H_inv_1_4, lnv)
print("\nAjuste lineal:")
print(f"Pendiente = {regresion.slope:.4f} ± {regresion.stderr:.4f}")
print(f"Intercepto = {regresion.intercept:.4f} ± {regresion.intercept_stderr:.4f}")
print(f"R^2 = {regresion.rvalue**2:.4f}, p = {regresion.pvalue:.2e}")

# Plot
plt.figure(figsize=(6, 4.5))
plt.errorbar(H_inv_1_4, lnv, yerr=err, fmt='o', capsize=4, label='Datos', color='navy')

x_fit = np.linspace(H_inv_1_4.min(), H_inv_1_4.max(), 200)
plt.plot(x_fit, regresion.slope * x_fit + regresion.intercept, '--', color='blue', label='Ajuste lineal')

plt.xlabel(r'$H^{-1/4}$ [$\mathrm{Oe}^{-1/4}$]')
plt.ylabel(r'$\ln(v)$ [ln(m/s)]')
plt.grid(True, linestyle=':')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("lnv_vs_Hmagnetico.pdf")
plt.show()

#%%

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours

# Parámetros
RAIZ = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe"
CAMPO = 140
TIEMPO_MS = 5
FRAME_IDX = 5
ESCALA = 0.8 / 1e6  # micrones/pixel
CENTRO = (153, 160)  # (x, y)

# Función para radio efectivo
def estimar_radio_suave(binary_image, centro):
    contours = find_contours(binary_image.astype(float), level=0.5)
    if not contours:
        return 0
    contorno_principal = max(contours, key=len)
    dx = contorno_principal[:, 1] - centro[0]
    dy = contorno_principal[:, 0] - centro[1]
    radios = np.sqrt(dx**2 + dy**2)
    return np.mean(radios)

# Cargar imagen
carpeta = os.path.join(RAIZ, f"{CAMPO:03d}")
nombre_archivo = f"Bin-P8137-{CAMPO:03d}Oe-{TIEMPO_MS}ms-{FRAME_IDX}.tif"
ruta = os.path.join(carpeta, nombre_archivo)

if not os.path.exists(ruta):
    raise FileNotFoundError(f"No se encontró la imagen: {ruta}")

imagen = imread(ruta)
R_suave = estimar_radio_suave(imagen, CENTRO)

# Graficar
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(imagen, cmap='gray')

# Circunferencia punteada
circ = plt.Circle(CENTRO, R_suave, color='red', linestyle='--', fill=False, linewidth=2)
ax.add_patch(circ)

# Línea del radio
x0, y0 = CENTRO
x1 = x0 + R_suave * np.cos(0)  # horizontal
y1 = y0 + R_suave * np.sin(0)
ax.plot([x0, x1], [y0, y1], color='red', linewidth=1.5)

# Punto del centro
ax.plot(x0, y0, 'ro')

# Texto r_eff
ax.text(x0 + R_suave / 2 + 5, y0 - 10, r"$r_\mathrm{eff}$", color='red', fontsize=14)

# Ajustes estéticos
ax.axis('off')
plt.tight_layout()

# Guardar la figura (descomentá para guardar)
plt.savefig("radio_efectivo.pdf", dpi=300)

plt.show()

#%%

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import find_contours
from scipy.stats import linregress

# Parámetros
RAIZ = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe"
CAMPO = 200
TIEMPO_MS = 0.5  # tiempo entre frames
ESCALA = 0.8 / 1e6  # micrones por pixel
CENTRO = (153, 160)

def estimar_radio_suave(imagen, centro):
    contornos = find_contours(imagen.astype(float), level=0.5)
    if not contornos:
        return 0
    c = max(contornos, key=len)
    dx = c[:, 1] - centro[0]
    dy = c[:, 0] - centro[1]
    return np.mean(np.sqrt(dx**2 + dy**2))

# Cargar radios para el campo dado
radios = []
for i in range(6):  # frames 0 a 5
    nombre = f"Bin-P8137-{CAMPO:03d}Oe-{TIEMPO_MS}ms-{i}.tif"
    ruta = os.path.join(RAIZ, f"{CAMPO:03d}", nombre)
    if os.path.exists(ruta):
        imagen = imread(ruta)
        r = estimar_radio_suave(imagen, CENTRO)
        radios.append(r)
    else:
        print(f"Falta imagen: {ruta}")

# Calcular ΔR, Δt
radios = np.array(radios)
delta_R = radios[1:] - radios[:-1]  # en pixeles
delta_t = np.full_like(delta_R, TIEMPO_MS / 1000)  # en segundos

# Convertir ΔR a micrones
delta_R_um = delta_R * ESCALA

# Ajuste lineal ΔR vs Δt
slope, intercept, r_value, _, _ = linregress(delta_t, delta_R_um)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(delta_t, delta_R_um, 'o', label='Datos')
plt.plot(delta_t, slope * delta_t + intercept, '--', label=f'Ajuste: $v = {slope:.2f}$ μm/s')
plt.xlabel(r'$\Delta t$ [s]')
plt.ylabel(r'$\Delta R$ [$\mu$m]')
plt.title(f'Cálculo de velocidad por diferencia de radios - {CAMPO} Oe')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

crop = (slice(340, 585), slice(900, 1150))
#%%

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from Analisis_poco_contorno import ImageEnhancer  # asegurate de que esté en el mismo directorio

# Configuración
N = 5
CAMPO_PABLO = 200
CAMPO_NUESTRO = 250
DT_PABLO = 0.5  # ms entre frames
DT_NUESTRO = 10  # ms por pulso
PULSOS_ENTRE_IMAGENES = 5
T_PULSO_NUESTRO = PULSOS_ENTRE_IMAGENES * DT_NUESTRO  # tiempo efectivo entre imágenes
crop = (slice(340, 585), slice(900, 1150))

# Rutas locales
RAIZ_PABLO = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe\200"
RAIZ_NUESTRA = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe\23-06-2025\250 x 10"

# Cargar imágenes Pablo (binarizadas)
imagenes_pablo = [
    imread(os.path.join(RAIZ_PABLO, f"Bin-P8137-200Oe-0.5ms-{i}.tif")) for i in range(N)
]

# Cargar nuestras imágenes (ordenadas por nombre, tomar 5 entre la 10 y la última)
archivos_nuestros = sorted([
    f for f in os.listdir(RAIZ_NUESTRA)
    if f.endswith(".tif") and "resta" in f
])
indices = np.linspace(10, len(archivos_nuestros) - 1, N, dtype=int)
archivos_seleccionados = [archivos_nuestros[i] for i in indices]

# Procesar nuestras imágenes
imagenes_nuestra_bin = []
for nombre in archivos_seleccionados:
    path = os.path.join(RAIZ_NUESTRA, nombre)
    im = imread(path)[crop]
    enhancer = ImageEnhancer(im)
    binary, _, _ = enhancer.procesar(
        suavizado=3,
        percentil_contornos=99.9,
        min_dist_picos=8000,
        metodo_contorno="binarizacion",
        mostrar=False
    )
    imagenes_nuestra_bin.append(binary)
#%%

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, N, figsize=(14, 3.5), gridspec_kw={'hspace': 0, 'wspace': 0})

for i in range(N):
    axs[0, i].imshow(imagenes_pablo[i], cmap='gray')
    axs[1, i].imshow(imagenes_nuestra_bin[i], cmap='gray')
    axs[0, i].axis('off')
    axs[1, i].axis('off')

# Título para fila de Pablo (centrado arriba de la fila 0)
fig.text(0.23, 0.89, r"$\mathbf{FO}$  -  $H = %d$ Oe, $\tau = %.1f$ ms" % (CAMPO_PABLO, DT_PABLO),
         ha='center', va='center', fontsize=11)

# Título para fila de nuestra muestra (centrado arriba de la fila 1)
fig.text(0.23, 0.48, r"$\mathbf{FI25}$  -  $H = %d$ Oe, $\tau = %d$ ms" % (CAMPO_NUESTRO, T_PULSO_NUESTRO),
         ha='center', va='center', fontsize=11)

plt.savefig("comparacion_dominios_horizontal.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Ruta a tus imágenes
ruta_antes = r'C:\Users\Tomas\Desktop\FACULTAD\LABO 6\comparacion imagenes\resta_20250414_135922.tif'    # ← cambiá esto
ruta_despues = r'C:\Users\Tomas\Desktop\FACULTAD\LABO 6\comparacion imagenes\resta_20250623_120939.tif' # ← y esto

# Cargar imágenes
img1 = mpimg.imread(ruta_antes)
img2 = mpimg.imread(ruta_despues)

# Crear figura horizontal
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Mostrar "antes"
ax[0].imshow(img1, cmap='gray')
ax[0].axis('off')
ax[0].set_title("Antes de ajustes", fontsize=14)

# Mostrar "después"
ax[1].imshow(img2, cmap='gray')
ax[1].axis('off')
ax[1].set_title("Después de ajustes", fontsize=14)

# Agregar flecha horizontal entre ambas
#fig.text(0.495, 0.5, r'$\Rightarrow$', fontsize=24, ha='center', va='center')

# Guardar figura
plt.tight_layout()
plt.savefig("comparacion_shutter_ajustes.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.exposure import rescale_intensity

# Cargar imagen y perfil
ruta = r'C:\Users\Tomas\Desktop\FACULTAD\LABO 6\comparacion imagenes\calibration_grid.jpg'
im = imread(ruta, as_gray=True)
im = rescale_intensity(im, out_range=(0, 1))
perfil = im[150, :]

# Binarizar con umbral 0.1
binario = np.where(perfil >= 0.09, 1, 0)

# Detectar bordes descendentes (1 → 0)
bordes = np.where((binario[:-1] == 1) & (binario[1:] == 0))[0]

# Calcular diferencias entre bordes (ancho de los cuadrados)
anchos = np.diff(bordes)

# Eliminar valores fuera de rango esperable (por si hay ruido)
anchos_filtrados = anchos[anchos > 5]  # ajustable si hay ruidos muy finos

# Calcular estadísticos
media = np.mean(anchos_filtrados)
std = np.std(anchos_filtrados, ddof=1)
N = len(anchos_filtrados)
error_media = std / np.sqrt(N)

# Escala
tam_cuadro_um = 60  # micrones
escala = tam_cuadro_um / media
error_escala = tam_cuadro_um * error_media / (media**2)

# Mostrar resultados
print(f"Anchos detectados: {anchos_filtrados}")
print(f"Ancho medio de los cuadrados: {media:.2f} ± {error_media:.2f} píxeles")
print(f"Escala estimada: {escala:.3f} ± {error_escala:.3f} µm/píxel")

# Visualización
plt.figure(figsize=(10, 4))
plt.plot(perfil, label="Perfil original", color='black')
plt.plot(binario, label="Perfil binarizado", linestyle='--', color='gray')
plt.vlines(bordes, ymin=0, ymax=1, color='red', linestyles='dotted', label='Bordes 1→0')
plt.title("Perfil de intensidad y detección de bordes")
plt.xlabel("Pixel")
plt.ylabel("Intensidad")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

import numpy as np

anchos_filtrados = np.array([170, 180])
media = np.mean(anchos_filtrados)
std = np.std(anchos_filtrados, ddof=1)
N = len(anchos_filtrados)
error_media = std / np.sqrt(N)

# Longitud física y su error
L_um = 60        # micrones
delta_L = 2      # error estimado (puede ajustarse)

# Escala y propagación de error total
escala = L_um / media
error_escala = escala * np.sqrt((delta_L / L_um)**2 + (error_media / media)**2)

# Mostrar resultados
print(f"Anchos detectados: {anchos_filtrados}")
print(f"Ancho medio: {media:.2f} ± {error_media:.2f} píxeles")
print(f"Escala estimada: {escala:.3f} ± {error_escala:.3f} µm/píxel")
