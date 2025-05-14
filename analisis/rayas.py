import os
import numpy as np
import datetime
from lucam import Lucam, API
import matplotlib.pyplot as plt
import glob
import csv
from numpy.fft import fft2, fftshift

camera = Lucam()
camera.CameraClose()
camera = Lucam()
frameformat, fps = camera.GetFormat()
frameformat.pixelFormat = API.LUCAM_PF_16
camera.SetFormat(frameformat, fps)
print("[INFO] Lucam camera initialized.")

# Set up output directory
output_dir = r'C:\Users\Marina\Documents\Labo 6\LuCam-app\test\test_estadistica\fotos_cable_nuevo'
os.makedirs(output_dir, exist_ok=True)

# Number of images
N = 400

print(f"[INFO] Capturing {N} 16-bit images and saving as float matrices...")

for i in range(N):
    img = camera.TakeSnapshot()
    if img is None:
        print(f"[WARNING] Snapshot {i} returned None. Skipping.")
        continue

    img_float = img.astype(np.float32)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(output_dir, f"snapshot_{i:04d}_{timestamp}.npy")
    np.save(filename, img_float)

    if (i + 1) % 100 == 0:
        print(f"[INFO] Saved {i+1} snapshots...")

print("[DONE] All snapshots saved.")


#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
from numpy.fft import fft2, fftshift

# === Configuración ===
input_folder = r"C:\Users\Marina\Documents\Labo 6\LuCam-app\test\test_estadistica\fotos_cable_nuevo"
output_folder = os.path.join(input_folder, "analysis_output_cablenuevo")
os.makedirs(output_folder, exist_ok=True)

paths = sorted(glob.glob(os.path.join(input_folder, "*.npy")))
if len(paths) == 0:
    raise RuntimeError(f"No se encontraron archivos .npy en {input_folder}")

num_images = min(1000, len(paths))  # hasta 1000 imágenes

# === Función para corrección local (media por fila propia) ===
def remove_lines_local(img):
    row_means = np.mean(img, axis=1, keepdims=True)
    corrected = img - row_means
    corrected += np.mean(img)  # preserva brillo global
    return corrected

# === Inicialización ===
means, stds, cvs = [], [], []
mean_image = None
first_image = None
row_profiles = []

corrected_dir = os.path.join(output_folder, "corrected_images_local")
os.makedirs(corrected_dir, exist_ok=True)

# === Procesar imágenes ===
for i, path in enumerate(paths[:num_images]):
    img = np.load(path)
    if first_image is None:
        first_image = img.copy()

    corrected = remove_lines_local(img)
    np.save(os.path.join(corrected_dir, f"corrected_{i:04d}.npy"), corrected)

    # Estadísticas
    m = np.mean(img)
    s = np.std(img)
    means.append(m)
    stds.append(s)
    cvs.append(s / m if m > 0 else 0)

    # Sumar para imagen media
    if mean_image is None:
        mean_image = img.astype(np.float64)
    else:
        mean_image += img

    # Perfiles por fila (original)
    row_mean = np.mean(img, axis=1)
    row_profiles.append(row_mean)

    # Guardar visualizaciones de las primeras 5 imágenes
    if i < 5:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        plt.title("Original")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(corrected, cmap='gray', vmin=np.percentile(corrected, 1), vmax=np.percentile(corrected, 99))
        plt.title("Corregida")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(corrected_dir, f"comparison_{i:04d}.png"))
        plt.close()

        # Histograma
        plt.figure()
        plt.hist(img.ravel(), bins=512, alpha=0.6, label="Original")
        plt.hist(corrected.ravel(), bins=512, alpha=0.6, label="Corregida")
        plt.title("Histograma de Intensidades")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(corrected_dir, f"histogram_{i:04d}.png"))
        plt.close()

        # Perfiles de fila y columna
        row_orig = np.mean(img, axis=1)
        row_corr = np.mean(corrected, axis=1)
        col_orig = np.mean(img, axis=0)
        col_corr = np.mean(corrected, axis=0)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(row_orig, label="Original")
        plt.plot(row_corr, label="Corregida")
        plt.title("Perfil de Fila")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(col_orig, label="Original")
        plt.plot(col_corr, label="Corregida")
        plt.title("Perfil de Columna")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(corrected_dir, f"profiles_{i:04d}.png"))
        plt.close()

# === Postprocesamiento seguro ===
if mean_image is not None:
    mean_image /= num_images
    global_mean = np.mean(mean_image)
    fpn = mean_image - global_mean
    fpn_std = np.std(fpn)
else:
    print("[WARNING] No se pudo calcular mean_image. No se procesaron imágenes.")
    mean_image = np.zeros_like(first_image, dtype=np.float64)
    global_mean = 0
    fpn = np.zeros_like(first_image)
    fpn_std = 0

row_profiles = np.array(row_profiles)
mean_row_profile = np.mean(row_profiles, axis=0)
row_var_profile = np.var(row_profiles, axis=0)

# === Gráficos globales ===
plt.figure()
plt.plot(means, label="Mean")
plt.plot(stds, label="Std")
plt.plot(cvs, label="CV")
plt.legend()
plt.title("Mean / Std / CV Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "mean_std_cv.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(mean_row_profile)
plt.title("Perfil Promedio por Fila")

plt.subplot(2, 1, 2)
plt.plot(row_var_profile)
plt.title("Varianza por Fila")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "perfil_fila_varianza.png"))
plt.close()

# === FFT de la primera imagen ===
img_fft = first_image - np.mean(first_image)
f = fft2(img_fft)
fshift = fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

plt.figure()
plt.imshow(magnitude_spectrum, cmap='inferno')
plt.title("FFT (primera imagen)")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "fft_first_image.png"))
plt.close()

# === FPN ===
plt.figure()
plt.imshow(fpn, cmap='seismic', vmin=-100, vmax=100)
plt.title(f"Fixed Pattern Noise (std = {fpn_std:.2f})")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "fpn_map.png"))
plt.close()

# === CSV resumen ===
summary_csv_path = os.path.join(output_folder, "summary_metrics.csv")
with open(summary_csv_path, mode="w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["# Images", num_images])
    writer.writerow(["Image Shape", f"{first_image.shape[0]}x{first_image.shape[1]}"])
    writer.writerow(["Global Mean", f"{global_mean:.2f}"])
    writer.writerow(["FPN Std", f"{fpn_std:.2f}"])
    writer.writerow(["Mean Row Std", f"{np.std(mean_row_profile):.2f}"])
    writer.writerow(["Mean Col Std", f"{np.std(np.mean(first_image, axis=0)):.2f}"])
    writer.writerow(["Mean of CVs", f"{np.mean(cvs):.4f}"])

print(f"[DONE] Corrección y análisis completados.\nResultados en:\n{output_folder}")


#%%
'''
stackear 100 imagenes y calcular la desviacion estanadar. hacer 
una imagen ploteando las desviaciones estandar
'''

# === Stack 100 images and compute per-pixel mean and std ===
stack_count = 100
stack_images = []

for path in paths[:stack_count]:
    img = np.load(path)
    stack_images.append(img)

stack_array = np.stack(stack_images, axis=0)  # shape: (N, H, W)
mean_stack = np.mean(stack_array, axis=0)
std_stack = np.std(stack_array, axis=0)

# === Plot: Per-pixel Mean ===
plt.figure()
plt.imshow(mean_stack, cmap='gray', vmin=np.percentile(mean_stack, 1), vmax=np.percentile(mean_stack, 99))
plt.title("Per-pixel Mean (First 100 Images)")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "mean_image_100stack.png"))
plt.close()

# === Plot: Per-pixel Std ===
plt.figure()
plt.imshow(std_stack, cmap='hot', vmin=0, vmax=np.percentile(std_stack, 99))
plt.title("Per-pixel Std Dev (First 100 Images)")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "std_image_100stack.png"))
plt.close()

#%%

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === Configuración ===
input_folder = r"C:\Users\Marina\Documents\Labo 6\LuCam-app\test\test_estadistica\fotos_cable_nuevo"
output_folder = os.path.join(input_folder, "line_correction_test")
os.makedirs(output_folder, exist_ok=True)

# === Cargar algunas imágenes ===
paths = sorted(glob.glob(os.path.join(input_folder, "*.npy")))
paths = paths[:100]  # usar primeras 100 imágenes

# === Función para corregir líneas ===
def remove_horizontal_lines(img):
    row_means = np.mean(img, axis=1, keepdims=True)
    corrected = img - row_means
    corrected += np.mean(img)  # conservar brillo original
    return corrected

# === Procesar y guardar comparaciones ===
for i, path in enumerate(paths[:5]):  # solo las primeras 5 para comparar visualmente
    img = np.load(path)
    corrected = remove_horizontal_lines(img)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(corrected, cmap='gray', vmin=np.percentile(corrected, 1), vmax=np.percentile(corrected, 99))
    plt.title("Corregida (media por fila)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"line_removed_{i:02d}.png"))
    plt.close()

#%%

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === Configuración ===
input_folder = r"C:\Users\Marina\Documents\Labo 6\LuCam-app\test\test_estadistica\fotos_cable_nuevo"
output_folder = os.path.join(input_folder, "line_removal_local_analysis")
os.makedirs(output_folder, exist_ok=True)

# === Cargar rutas ===
paths = sorted(glob.glob(os.path.join(input_folder, "*.npy")))
paths = paths[:5]  # analizar primeras 5 imágenes

# === Función de corrección por media por fila de sí misma ===
def remove_horizontal_lines_local(img):
    row_means = np.mean(img, axis=1, keepdims=True)
    corrected = img - row_means
    corrected += np.mean(img)  # conservar brillo promedio
    return corrected

# === Procesar imágenes ===
for i, path in enumerate(paths):
    img = np.load(path)
    corrected = remove_horizontal_lines_local(img)

    # === Histogramas ===
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(img.ravel(), bins=256, alpha=0.7, label='Original', color='gray')
    plt.hist(corrected.ravel(), bins=256, alpha=0.7, label='Corregida', color='red')
    plt.title("Histogramas")
    plt.legend()
    plt.grid(True)

    # === Imágenes ===
    plt.subplot(1, 2, 2)
    plt.imshow(corrected, cmap='gray', vmin=np.percentile(corrected, 1), vmax=np.percentile(corrected, 99))
    plt.title("Imagen Corregida")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"histogram_and_corrected_{i:02d}.png"))
    plt.close()

    # === Perfiles de Intensidad ===
    row_profile_orig = np.mean(img, axis=1)
    row_profile_corr = np.mean(corrected, axis=1)
    col_profile_orig = np.mean(img, axis=0)
    col_profile_corr = np.mean(corrected, axis=0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(row_profile_orig, label='Original')
    plt.plot(row_profile_corr, label='Corregida')
    plt.title("Perfil de Fila (media por fila)")
    plt.xlabel("Fila")
    plt.ylabel("Intensidad promedio")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(col_profile_orig, label='Original')
    plt.plot(col_profile_corr, label='Corregida')
    plt.title("Perfil de Columna (media por columna)")
    plt.xlabel("Columna")
    plt.ylabel("Intensidad promedio")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"profiles_{i:02d}.png"))
    plt.close()
    
#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# === Paths ===
base_path = r"C:\Users\Marina\Documents\Labo 6\LuCam-app\test\test_estadistica"
folder_viejo = os.path.join(base_path, "fotos_cable_viejo")
folder_nuevo = os.path.join(base_path, "fotos_cable_nuevo")

paths_viejo = sorted(glob.glob(os.path.join(folder_viejo, "*.npy")))
paths_nuevo = sorted(glob.glob(os.path.join(folder_nuevo, "*.npy")))

# === Chequeo mínimo ===
assert len(paths_viejo) > 0, "No se encontraron imágenes en fotos_cable_viejo"
assert len(paths_nuevo) > 0, "No se encontraron imágenes en fotos_cable_nuevo"

# === Calcular perfiles promedio por fila con std ===
def compute_row_profile_stats(paths, N=100):
    profiles = []
    for path in paths[:N]:
        img = np.load(path)
        row_profile = np.mean(img, axis=1)
        profiles.append(row_profile)
    profiles = np.array(profiles)
    mean_profile = np.mean(profiles, axis=0)
    std_profile = np.std(profiles, axis=0)
    return mean_profile, std_profile

mean_viejo, std_viejo = compute_row_profile_stats(paths_viejo)
mean_nuevo, std_nuevo = compute_row_profile_stats(paths_nuevo)

# === Plot: perfil de intensidad promedio por fila con desvío ===
plt.figure(figsize=(10, 6))
rows = np.arange(len(mean_viejo))

plt.plot(rows, mean_viejo, label="Viejo - media", color='blue')
plt.fill_between(rows, mean_viejo - std_viejo, mean_viejo + std_viejo, color='blue', alpha=0.3)

plt.plot(rows, mean_nuevo, label="Nuevo - media", color='green')
plt.fill_between(rows, mean_nuevo - std_nuevo, mean_nuevo + std_nuevo, color='green', alpha=0.3)

plt.xlabel("Fila")
plt.ylabel("Intensidad promedio")
plt.title("Perfiles de Intensidad Promedio por Fila (Viejo vs Nuevo)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "perfil_fila_viejo_vs_nuevo.png"))
plt.show()

# === Comparar primeras imágenes lado a lado ===
img_viejo = np.load(paths_viejo[0])
img_nuevo = np.load(paths_nuevo[0])

vmin = min(np.percentile(img_viejo, 1), np.percentile(img_nuevo, 1))
vmax = max(np.percentile(img_viejo, 99), np.percentile(img_nuevo, 99))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_viejo, cmap='gray', vmin=vmin, vmax=vmax)
plt.title("Primera Imagen - Cable Viejo")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(img_nuevo, cmap='gray', vmin=vmin, vmax=vmax)
plt.title("Primera Imagen - Cable Nuevo")
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(base_path, "primeras_imagenes_viejo_vs_nuevo.png"))
plt.show()

# === Histogramas de la primera imagen ===
plt.figure(figsize=(10, 5))
plt.hist(img_viejo.ravel(), bins=512, alpha=0.5, label="Cable Viejo", color='blue')
plt.hist(img_nuevo.ravel(), bins=512, alpha=0.5, label="Cable Nuevo", color='green')
plt.title("Histogramas de la Primera Imagen")
plt.xlabel("Intensidad")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "histogramas_viejo_vs_nuevo.png"))
plt.show()
