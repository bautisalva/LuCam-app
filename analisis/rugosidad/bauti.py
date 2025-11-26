# python
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import measure

def _ensure_closed_no_dup(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    if C.shape[0] < 2:
        raise ValueError("La curva debe tener al menos 2 puntos.")
    if np.allclose(C[0], C[-1]):
        return C[:-1].copy()
    return C.copy()

def _largest_contour_yx(mask: np.ndarray) -> np.ndarray:
    conts = measure.find_contours(mask.astype(float), level=0.5)
    if not conts:
        raise ValueError("No se encontró ningún contorno en la máscara.")
    contour = max(conts, key=len)
    return _ensure_closed_no_dup(contour)

def _resample_by_arclength(contour_yx: np.ndarray, N: int) -> np.ndarray:
    C = _ensure_closed_no_dup(contour_yx)
    d = np.linalg.norm(np.diff(np.vstack([C, C[0]]), axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(d)))
    L = s[-1]
    st = np.linspace(0, L, N, endpoint=False)
    y = np.interp(st, s, np.r_[C[:, 0], C[0, 0]])
    x = np.interp(st, s, np.r_[C[:, 1], C[0, 1]])
    return np.column_stack([y, x])

def compute_arc_chord_stats(x: np.ndarray,
                            y: np.ndarray,
                            d_values: np.ndarray,
                            delta_d: float | None = None,
                            closed: bool = True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x e y deben tener la misma forma.")
    N = x.size
    if N < 2:
        raise ValueError("El contorno debe tener al menos 2 puntos.")

    pts = np.column_stack([x, y])
    diffs_seg = np.diff(np.vstack([pts, pts[0]]), axis=0)
    seg_lengths = np.sqrt(np.sum(diffs_seg**2, axis=1))
    perimetro = float(np.sum(seg_lengths))
    ds = perimetro / N

    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.nan)

    idx_i, idx_j = np.triu_indices(N, k=1)
    dist_flat = dist[idx_i, idx_j]

    delta_idx = np.abs(idx_j - idx_i)
    if closed:
        delta_idx = np.minimum(delta_idx, N - delta_idx)
    arc_flat = ds * delta_idx.astype(float)

    d_values = np.asarray(d_values, dtype=float)
    Nd = d_values.size
    if Nd < 1:
        raise ValueError("d_values debe tener al menos un valor.")

    if delta_d is None:
        if Nd > 1:
            step = np.min(np.diff(np.sort(d_values)))
            delta_d = step
        else:
            d_max = np.nanmax(dist_flat)
            delta_d = 0.05 * d_max

    mean_arc = np.full(Nd, np.nan)
    counts = np.zeros(Nd, dtype=int)

    for k, d0 in enumerate(d_values):
        d_min = d0 - 0.5 * delta_d
        d_max = d0 + 0.5 * delta_d
        m = (dist_flat >= d_min) & (dist_flat < d_max)
        if np.any(m):
            mean_arc[k] = np.nanmean(arc_flat[m])
            counts[k] = int(np.sum(m))

    return mean_arc, counts, perimetro, ds

def parse_field_oe_from_folder(folder_name: str) -> int | None:
    m = re.search(r'(\d+)\s*Oe', folder_name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def list_images(folder_path: str) -> list[str]:
    exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
    files = []
    for fname in os.listdir(folder_path):
        _, ext = os.path.splitext(fname)
        if ext.lower() in exts:
            files.append(os.path.join(folder_path, fname))
    return sorted(files)

def compute_roughness_slope_for_image(img_path: str, Msamples: int = 2048) -> float:
    # Calcula ⟨s⟩(d), ajusta log10(⟨s⟩) vs log10(d) con recta y devuelve la pendiente.
    mask = imread(img_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_bool = mask > 0
    contour_yx = _largest_contour_yx(mask_bool)
    Ceq = _resample_by_arclength(contour_yx, Msamples)
    y = Ceq[:, 0]
    x = Ceq[:, 1]

    pts = np.column_stack([x, y])
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.nan)
    d_max = np.nanmax(dist)

    diffs_seg = np.diff(np.vstack([pts, pts[0]]), axis=0)
    seg_lengths = np.sqrt(np.sum(diffs_seg**2, axis=1))
    perimetro = float(np.sum(seg_lengths))
    ds = perimetro / Msamples

    d_min = 2.0 * ds
    Nd = 100
    d_values = np.linspace(d_min, d_max, Nd)

    mean_arc, _, _, _ = compute_arc_chord_stats(x, y, d_values, delta_d=None, closed=True)

    valid = (~np.isnan(mean_arc)) & (mean_arc > 0) & (d_values > 0)
    if np.count_nonzero(valid) < 2:
        return np.nan

    log_d = np.log10(d_values[valid])
    log_s = np.log10(mean_arc[valid])

    # Ajuste lineal en espacio log-log: log(s) = a * log(d) + b
    a, b = np.polyfit(log_d, log_s, 1)
    return float(a)

def main():
    base_dir = r'D:\MEDICIONES L7\Imagenes ya binarizadas'
    Msamples = 2048

    campos = []
    rugosidad_promedio = []

    # Carpetas de pulsos cuadrados: empiezan con 'Cuadrado' y contienen 'Oe'
    for entry in os.scandir(base_dir):
        if not entry.is_dir():
            continue
        fname = entry.name
        if not fname.lower().startswith('cuadrado') or 'oe' not in fname.lower():
            continue

        campo = parse_field_oe_from_folder(fname)
        if campo is None:
            continue

        img_paths = list_images(entry.path)
        if len(img_paths) == 0:
            print(f'Sin imágenes en carpeta `{entry.path}`')
            continue

        slopes = []
        for img_path in img_paths:
            try:
                slope = compute_roughness_slope_for_image(img_path, Msamples=Msamples)
                if not np.isnan(slope):
                    slopes.append(slope)
            except Exception as e:
                print(f'Error procesando `{img_path}`: {e}')

        if len(slopes) == 0:
            print(f'Sin rugosidades válidas en `{entry.path}`')
            continue

        R_folder = float(np.mean(slopes))
        campos.append(campo)
        rugosidad_promedio.append(R_folder)
        print(f'Carpeta: `{entry.path}` | Campo: {campo} Oe | Imágenes: {len(slopes)} | Rugosidad (pendiente) promedio: {R_folder:.4f}')

    if len(campos) == 0:
        print('No se encontraron datos válidos.')
        return

    campos = np.array(campos)
    rugosidad_promedio = np.array(rugosidad_promedio)
    order = np.argsort(campos)
    campos = campos[order]
    rugosidad_promedio = rugosidad_promedio[order]

    plt.figure(figsize=(6, 5))
    plt.plot(campos, rugosidad_promedio, 'o-')
    plt.xlabel('Campo aplicado [Oe]')
    plt.ylabel('Rugosidad (pendiente log-log de ⟨s⟩ vs d)')
    plt.title('Rugosidad promedio vs Campo aplicado (pulsos cuadrados)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()