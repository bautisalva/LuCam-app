from __future__ import annotations
import numpy as np
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import measure


def _ensure_closed_no_dup(C: np.ndarray) -> np.ndarray:
    """
    Asegura que la curva esté cerrada y evita duplicar el primer punto al final.
    C se espera como array de forma (N, 2) con columnas [y, x].
    """
    C = np.asarray(C, dtype=float)
    if C.shape[0] < 2:
        raise ValueError("La curva debe tener al menos 2 puntos.")
    if np.allclose(C[0], C[-1]):
        return C[:-1].copy()
    return C.copy()


def _largest_contour_yx(mask: np.ndarray) -> np.ndarray:
    """
    Devuelve el contorno más grande (en cantidad de puntos) de la máscara binaria.
    Coordenadas en formato (y, x) como devuelve skimage.measure.find_contours.
    """
    conts = measure.find_contours(mask.astype(float), level=0.5)
    if not conts:
        raise ValueError("No se encontró ningún contorno en la máscara.")
    contour = max(conts, key=len)  # (N, 2) en (y, x)
    return _ensure_closed_no_dup(contour)


def _resample_by_arclength(contour_yx: np.ndarray, N: int) -> np.ndarray:
    """
    Reparametriza el contorno (y, x) a N puntos equiespaciados en longitud de arco.
    """
    C = _ensure_closed_no_dup(contour_yx)
    # Distancias entre puntos consecutivos (cerrando la curva)
    d = np.linalg.norm(np.diff(np.vstack([C, C[0]]), axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(d)))
    L = s[-1]
    st = np.linspace(0, L, N, endpoint=False)
    y = np.interp(st, s, np.r_[C[:, 0], C[0, 0]])
    x = np.interp(st, s, np.r_[C[:, 1], C[0, 1]])
    return np.column_stack([y, x])  # (N, 2) en (y, x)


def compute_arc_chord_stats(x: np.ndarray,
                            y: np.ndarray,
                            d_values: np.ndarray,
                            delta_d: float | None = None,
                            closed: bool = True):
    """
    Calcula, para un contorno reparametrizado (x, y),
    la longitud de arco media entre todos los pares de puntos
    cuya distancia euclídea está en torno a d, para cada d en d_values.

    Parámetros
    ----------
    x, y : array_like, shape (N,)
        Coordenadas del contorno reparametrizado (puntos equiespaciados en arco).
    d_values : array_like, shape (Nd,)
        Valores de distancia euclídea (cuerda) en los que se evalúa.
    delta_d : float, opcional
        Ancho del bin alrededor de cada d (por defecto, el paso entre d_values).
    closed : bool, default True
        Si True, se asume contorno cerrado y se toma el arco mínimo
        entre los dos sentidos (horario / antihorario).

    Devuelve
    --------
    mean_arc : ndarray, shape (Nd,)
        Longitud de arco media para cada d (NaN si no hubo pares en ese bin).
    counts : ndarray, shape (Nd,)
        Cantidad de pares (i, j) usados en cada bin (con i < j).
    perimetro : float
        Perímetro total del contorno.
    ds : float
        Paso de arco medio (perímetro / N).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x e y deben tener la misma forma.")
    N = x.size
    if N < 2:
        raise ValueError("El contorno debe tener al menos 2 puntos.")

    pts = np.column_stack([x, y])  # (N, 2)

    # Distancias de cada segmento y perímetro
    diffs_seg = np.diff(np.vstack([pts, pts[0]]), axis=0)  # (N, 2)
    seg_lengths = np.sqrt(np.sum(diffs_seg**2, axis=1))    # (N,)
    perimetro = float(np.sum(seg_lengths))
    ds = perimetro / N

    # Matriz de distancias euclídeas
    diff = pts[:, None, :] - pts[None, :, :]       # (N, N, 2)
    dist = np.sqrt(np.sum(diff**2, axis=-1))       # (N, N)
    np.fill_diagonal(dist, np.nan)

    # Usamos solo pares i < j para no contar dos veces lo mismo
    idx_i, idx_j = np.triu_indices(N, k=1)
    dist_flat = dist[idx_i, idx_j]                # (N_pairs,)

    # Distancias de arco (en número de pasos) entre índices
    delta_idx = np.abs(idx_j - idx_i)
    if closed:
        delta_idx = np.minimum(delta_idx, N - delta_idx)
    arc_flat = ds * delta_idx.astype(float)       # (N_pairs,)

    # Preparar bins en d
    d_values = np.asarray(d_values, dtype=float)
    Nd = d_values.size
    if Nd < 1:
        raise ValueError("d_values debe tener al menos un valor.")

    if delta_d is None:
        if Nd > 1:
            step = np.min(np.diff(np.sort(d_values)))
            delta_d = step
        else:
            # Si solo hay un valor, tomamos 5% del máximo como ventana
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


def main():
    # ============================================================
    # Parámetros de entrada
    # ============================================================
    # >>> Cambiá esta ruta por la de tu imagen binarizada <<<
    ruta_imagen = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe\090\Bin-P8137-090Oe-50ms-4.tif"

    # Cantidad de puntos para el contorno reparametrizado
    Msamples = 2048

    # ============================================================
    # 1) Cargar máscara binarizada
    # ============================================================
    mask = imread(ruta_imagen)

    # Si viene RGB, nos quedamos con un canal (asumo imagen ya binarizada)
    if mask.ndim == 3:
        mask = mask[..., 0]

    mask_bool = mask > 0  # True = interior del dominio

    # ============================================================
    # 2) Sacar contorno más grande y reparametrizar por arclonga
    # ============================================================
    contour_yx = _largest_contour_yx(mask_bool)          # (Ny, 2) en (y, x)
    Ceq = _resample_by_arclength(contour_yx, Msamples)   # (Msamples, 2) en (y, x)

    y = Ceq[:, 0]
    x = Ceq[:, 1]

    # ============================================================
    # 3) Definir grilla de distancias d
    # ============================================================
    pts = np.column_stack([x, y])
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.nan)
    d_max = np.nanmax(dist)

    # Paso típico de arco ~ ds, tomamos d_min ~ 2 ds
    diffs_seg = np.diff(np.vstack([pts, pts[0]]), axis=0)
    seg_lengths = np.sqrt(np.sum(diffs_seg**2, axis=1))
    perimetro = float(np.sum(seg_lengths))
    ds = perimetro / Msamples

    d_min = 2.0 * ds
    Nd = 100 # cantidad de puntos en la grilla de d (podés subirlo si querés pasos más finos)
    d_values = np.linspace(d_min, d_max, Nd)

    # ============================================================
    # 4) Calcular longitud de arco media vs d
    # ============================================================
    mean_arc, counts, perimetro_calc, ds_calc = compute_arc_chord_stats(
        x, y, d_values, delta_d=None, closed=True
    )

    # ============================================================
    # 5) Plot ⟨s⟩(d) vs d
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(d_values, mean_arc, "o-")
    ax.set_xlabel("Distancia euclídea d (cuerda) [px]")
    ax.set_ylabel("Longitud de arco media ⟨s⟩(d) [px]")
    ax.set_title("Longitud de arco media entre puntos a distancia d")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
