
import os, re, csv, time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

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
        background = gaussian(
            self.image.astype(np.float32),
            sigma=self.sigma_background,
            preserve_range=True
        )
        return self.image.astype(np.float32) - self.alpha * background

    def _find_large_contours(self, binary, percentil_contornos=99.9):
        contours = find_contours(binary, level=0.5)
        if percentil_contornos > 0 and contours:
            def area_contour(c):
                x, y = c[:, 1], c[:, 0]
                return 0.5 * np.abs(
                    np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
                )
            areas = np.array([area_contour(c) for c in contours])
            umbral = np.percentile(areas, percentil_contornos)
            return [c for c, a in zip(contours, areas) if a >= umbral]
        return contours

    def procesar(self, suavizado=3, percentil_contornos=99.9):
        img = (
            self._subtract_background()
            if self.alpha > 0
            else self.image.astype(float)
        )
        smooth = uniform_filter(img, size=suavizado)
        th = threshold_otsu(smooth)
        binary = (smooth > th).astype(np.uint16)
        contours = self._find_large_contours(binary, percentil_contornos)
        return binary, contours


# ------------------------------------------------------------
# Cálculo de Var(u)
# ------------------------------------------------------------
def var_u(binary1, binary2, contours1):
    delta = binary2.astype(int) - binary1.astype(int)
    changed = np.argwhere(delta != 0)
    if changed.size == 0 or not contours1:
        return 0.0

    cont_points = np.vstack(contours1)
    P = perimeter(binary1)
    if P == 0:
        return 0.0

    tree = cKDTree(cont_points)
    dists, _ = tree.query(changed)

    sum_uprime = dists.sum()
    sum_abs_da = np.abs(delta).sum()

    return (2 * sum_uprime / P) - (sum_abs_da / P) ** 2


# ------------------------------------------------------------
# Cargar imágenes desde carpeta
# ------------------------------------------------------------
def load_images_folder(folder, regex_pattern, min_index=15):
    pattern = re.compile(regex_pattern)
    items = []
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            try:
                key = int(m.group(1))
            except ValueError:
                key = m.group(1)
            items.append((key, fname))
    items.sort(key=lambda x: x[0])
    items = items[min_index:]

    images, filenames = [], []
    for _, fname in items:
        try:
            img = imread(os.path.join(folder, fname))
            images.append(img)
            filenames.append(fname)
        except Exception:
            print(f"No se pudo cargar {fname}")
    return images, filenames


# ------------------------------------------------------------
# Guardar imagen con contornos (sin mostrar)
# ------------------------------------------------------------
def save_contour_overlay(image, contours, out_folder, base_name):
    os.makedirs(out_folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], '-r', linewidth=1)
    ax.axis('off')

    out_path = os.path.join(out_folder, f"{base_name}_contornos.png")
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------
def pipeline(folder,
             regex_pattern,
             already_binary=False,
             output_folder=None):
    images, filenames = load_images_folder(folder, regex_pattern, min_index=15)

    # Recortar ROI solo en imágenes que empiecen por 'resta_'
    for i, fname in enumerate(filenames):
        if fname.startswith("resta_"):
            images[i] = images[i][140:280, 345:495]

    # Carpeta de salida
    output_folder = output_folder or folder
    debug_folder = os.path.join(output_folder, "debug_contornos")
    os.makedirs(debug_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "pipeline_log.csv")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Paso", "Archivo", "ChangedPixels", "Perimetro",
            "MeanDist", "MaxDist", "Var_u", "Duracion(s)"
        ])

        variances = []
        for k in range(len(images) - 1):
            start = time.time()

            # Binarizado y contornos
            if already_binary:
                b1 = (images[k] > 0).astype(np.uint8)
                b2 = (images[k + 1] > 0).astype(np.uint8)
                contours = find_contours(b1, level=0.5)
            else:
                enh1 = ImageEnhancer(images[k], sigma_background=100, alpha=0)
                enh2 = ImageEnhancer(images[k + 1], sigma_background=100, alpha=0)
                b1, contours = enh1.procesar()
                b2, _ = enh2.procesar()

            if b1.shape != b2.shape:
                raise ValueError(f"Shapes mismatch en paso {k}: {b1.shape} vs {b2.shape}")

            # Guardar overlay sin mostrar
            base = f"step_{k:03d}_{filenames[k].rsplit('.',1)[0]}"
            save_contour_overlay(b1, contours, debug_folder, base)

            # Métricas
            delta = b2.astype(int) - b1.astype(int)
            changed = np.argwhere(delta != 0)
            P = perimeter(b1)

            if contours:
                tree = cKDTree(np.vstack(contours))
                dists, _ = tree.query(changed)
            else:
                dists = np.array([])

            mean_d = float(dists.mean()) if dists.size else 0.0
            max_d = float(dists.max()) if dists.size else 0.0
            v = var_u(b1, b2, contours)
            variances.append(v)

            dur = time.time() - start
            writer.writerow([
                k, filenames[k], len(changed), P,
                f"{mean_d:.3f}", f"{max_d:.3f}", f"{v:.5f}",
                f"{dur:.2f}"
            ])

            if v < 0 or (len(variances) > 1 and v > 10 * np.median(variances[:-1])):
                print(f"Alerta paso {k}: Var(u) = {v:.5f} (atípico)")

    # Gráficos de dispersión guardados con nombre único
    for folder_path in [folder]:
        base = os.path.basename(folder_path).replace(" ", "_")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(len(variances)), variances, marker="o")
        ax.set_xlabel("Paso")
        ax.set_ylabel("Var(u)")
        ax.set_title(f"Var(u) en {base}")
        ax.grid(True)

        out_name = f"dispersion_varu_{base}.png"
        out_path = os.path.join(output_folder, out_name)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Gráfico de dispersión guardado en: {out_path}")

    return variances


# ------------------------------------------------------------
# Ejecución en carpetas
# ------------------------------------------------------------
if __name__ == "__main__":
    salida = r"C:\Users\Marina\Documents\Labo 6\imagenes\test_dispersion"

    base1 = r"C:\Users\Marina\Documents\Labo 6\imagenes\250 x 10"
    regex1 = r'resta_(\d{8}_\d{6})\.tif'
    pipeline(base1, regex1, already_binary=False, output_folder=salida)

    base2 = r"C:\Users\Marina\Documents\Labo 6\imagenes\test_velocidades"
    regex2 = r".*-(\d+)\.tif$"
    pipeline(base2, regex2, already_binary=True, output_folder=salida)