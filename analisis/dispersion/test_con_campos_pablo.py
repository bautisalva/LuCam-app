# -*- coding: utf-8 -*-
import os, re, glob
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.measure import find_contours
from scipy.ndimage import distance_transform_edt

# === IMPORTA TU CLASE (archivo provisto) ===
from Clase_dispersión_radio_medio import RadialGrowthAnalyzer

# ===================== CONFIG FO (igual a DISPERSION 1.py) =====================
ROOT_FO = r'C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe'  # <- ajustá si hace falta
ESCALA_UM_PX = 0.34  # µm/px (misma calibración)

CAMPOS_FO = [80, 90, 100, 110, 120, 130, 140, 150, 170, 180, 190, 200]
TIEMPOS_MS = {
    80: 100, 90: 50, 100: 30, 110: 20, 120: 10, 130: 10,
    140: 5, 150: 3, 170: 2, 180: 1, 190: 1, 200: 0.5,
    # 160: (no definido; se omite)
}

def zpad3(h: int) -> str:
    return f"{h:03d}"

def natural_sort_key(filename: str):
    """Ordena por número final o índice en nombre, como en tu loader."""
    base = os.path.basename(filename)
    m = re.search(r'(\d+)(?=\.\w+$)', base)  # número antes de la extensión
    if m:
        return (0, int(m.group(1)))
    # fallback: busca primer número en el nombre
    m2 = re.search(r'(\d+)', base)
    return (1, int(m2.group(1)) if m2 else 2, base)

def build_fo_filelist(root_fo: str):
    """Devuelve dict[H] -> lista de paths (ordenados) para ese campo."""
    cfg = {}
    for H in CAMPOS_FO:
        if H not in TIEMPOS_MS:  # por si falta 160
            continue
        sub = os.path.join(root_fo, zpad3(H))
        t = TIEMPOS_MS[H]
        t_str = ("%.3f" % t).rstrip('0').rstrip('.') if isinstance(t, float) else str(t)
        patt = f"Bin-P8137-{zpad3(H)}Oe-{t_str}ms-*.tif"
        files = glob.glob(os.path.join(sub, patt))
        files.sort(key=natural_sort_key)
        if files:
            cfg[H] = files
    return cfg

# ===================== MÉTODOS GEOMÉTRICOS =====================

def longest_contour(binary_img):
    conts = find_contours(binary_img.astype(float), level=0.5)
    return max(conts, key=lambda c: c.shape[0]) if conts else None

def contour_length(coords):
    if coords is None or len(coords) < 2:
        return 0.0
    y, x = coords[:, 0], coords[:, 1]
    dy = np.diff(y, append=y[0]); dx = np.diff(x, append=x[0])
    return float(np.sum(np.hypot(dx, dy)))

# ---- Cuantización de anillos con fix "touch_last" (off-by-one) ----
def _quantize_ring_index(d, eps=1e-6):
    # asigna anillos (k-1, k]  -> ceil(d - eps) - 1
    k = np.ceil(d - eps).astype(int) - 1
    return np.maximum(k, 0)

def bands_from_C1_to_C2(C1, C2):
    """Fajas C1→C2: growth=C2&~C1, shrink=C1&~C2 + índices de anillo k_out/k_in."""
    C1b = (C1.astype(np.uint8) > 0)
    C2b = (C2.astype(np.uint8) > 0)
    growth = C2b & (~C1b)   # expansión
    shrink = C1b & (~C2b)   # retracción

    d_in  = distance_transform_edt(C1b)    # adentro de C1
    d_out = distance_transform_edt(~C1b)   # afuera de C1

    k_out = np.full(C1b.shape, -1, dtype=int)
    k_in  = np.full(C1b.shape, -1, dtype=int)
    k_out[~C1b] = _quantize_ring_index(d_out[~C1b])
    k_in[C1b]   = _quantize_ring_index(d_in[C1b])

    return dict(C1b=C1b, C2b=C2b, growth=growth, shrink=shrink, k_out=k_out, k_in=k_in)

def ak_pk_by_ring_from_C1(bands):
    """Calcula Ak y Pk por faja: exterior (crecimiento) e interior (retracción)."""
    C1b, growth, shrink = bands["C1b"], bands["growth"], bands["shrink"]
    k_out, k_in = bands["k_out"], bands["k_in"]

    # Exterior
    max_ko = int(k_out[~C1b].max()) if (~C1b).any() else -1
    Lpos = max_ko + 1 if max_ko >= 0 else 0
    Ak_pos = np.zeros(Lpos, dtype=np.int64)
    Pk_pos = np.zeros(Lpos, dtype=np.int64)
    for k in range(Lpos):
        ring = (~C1b) & (k_out == k)      # faja de 1 px
        Pk_pos[k] = int(ring.sum())       # todos los puntos de la faja
        if Pk_pos[k] > 0:
            Ak_pos[k] = int((ring & growth).sum())  # los que CAMBIARON dentro de esa faja

    # Interior
    max_ki = int(k_in[C1b].max()) if (C1b).any() else -1
    Lneg = max_ki + 1 if max_ki >= 0 else 0
    Ak_neg = np.zeros(Lneg, dtype=np.int64)
    Pk_neg = np.zeros(Lneg, dtype=np.int64)
    for k in range(Lneg):
        ring = (C1b) & (k_in == k)
        Pk_neg[k] = int(ring.sum())
        if Pk_neg[k] > 0:
            Ak_neg[k] = int((ring & shrink).sum())

    rk_pos = np.arange(Lpos, dtype=float) + 0.5
    rk_neg = np.arange(Lneg, dtype=float) + 0.5
    return rk_pos, rk_neg, Ak_pos, Ak_neg, Pk_pos, Pk_neg

def perimeters_P1_P2(C1, C2):
    P1 = contour_length(longest_contour(C1))
    P2 = contour_length(longest_contour(C2))
    return P1, P2, 0.5*(P1+P2) if (P1+P2)>0 else np.nan

def dispersion_pk(C1, C2, use_curvature=True):
    """
    Tu método por fajas con Pk adentro.
    ū      = Σ (Ak_pos/Pk_pos) - Σ (Ak_neg/Pk_neg)
    <u²>    = 2 * Σ r_k (Ak/Pk)   [pos + neg]
              + (curv opcional)  - (4π/Pref) * Σ r_k^2 (Ak/Pk)
    Var     = <u²> - ū²  (sin truncar negativos)
    """
    bands = bands_from_C1_to_C2(C1, C2)
    rk_pos, rk_neg, Ak_pos, Ak_neg, Pk_pos, Pk_neg = ak_pk_by_ring_from_C1(bands)

    mpos = Pk_pos > 0
    mneg = Pk_neg > 0

    # u medio
    u_mean = float((Ak_pos[mpos]/Pk_pos[mpos]).sum() - (Ak_neg[mneg]/Pk_neg[mneg]).sum())

    # término base 2*Σ r_k (Ak/Pk)
    M1 = float((rk_pos[mpos] * (Ak_pos[mpos]/Pk_pos[mpos])).sum()
             + (rk_neg[mneg] * (Ak_neg[mneg]/Pk_neg[mneg])).sum())
    u2_mean = 2.0 * M1

    if use_curvature:
        # - (4π/Pref) * Σ r_k^2 (Ak/Pk)
        P1, P2, Pref = perimeters_P1_P2(C1, C2)
        M2 = float(((rk_pos[mpos]**2) * (Ak_pos[mpos]/Pk_pos[mpos])).sum()
                 + ((rk_neg[mneg]**2) * (Ak_neg[mneg]/Pk_neg[mneg])).sum())
        if np.isfinite(Pref) and Pref > 0:
            u2_mean += (-4.0*np.pi/Pref) * M2

    var_u = u2_mean - u_mean**2
    return u_mean, u2_mean, var_u

def dispersion_radial(contour1, contour2, n_theta=720, quantile=0.90):
    """Método de radio efectivo con tu clase."""
    rga = RadialGrowthAnalyzer(n_theta=n_theta, quantile=quantile)
    _, _, _, _, stats, _ = rga.analyze(contour1, contour2, center=None, plot=False)
    return stats['u_mean'], stats['u_var']

# ===================== RECORRIDO POR CAMPO Y PLOT =====================

def compute_series_for_field(files):
    """
    Dado el listado de imágenes de un campo (≈6),
    devuelve listas: var_pk_list, var_rad_list (una por par consecutivo).
    """
    var_pk_list = []
    var_rad_list = []

    for i in range(len(files)-1):
        im1 = imread(files[i])
        im2 = imread(files[i+1])
        # Asegurar binarios {0,1}
        b1 = (im1 > 0).astype(np.uint8)
        b2 = (im2 > 0).astype(np.uint8)

        # --- Tu método (fajas con Pk) ---
        _, _, var_pk = dispersion_pk(b1, b2, use_curvature=False)  # poné False si no querés curvatura
        var_pk_list.append(var_pk)

        # --- Radio efectivo ---
        c1 = longest_contour(b1)
        c2 = longest_contour(b2)
        if (c1 is not None) and (c2 is not None):
            _, var_rad = dispersion_radial(c1, c2, n_theta=720, quantile=0.90)
            var_rad_list.append(var_rad)
        else:
            var_rad_list.append(np.nan)

    return np.array(var_pk_list, float), np.array(var_rad_list, float)

def stats_by_field(var_series_dict):
    """
    var_series_dict: dict[H] -> np.array (var por par)
    devuelve arrays Hs, mean, std (en µm²)
    """
    Hs, means, stds = [], [], []
    for H in sorted(var_series_dict.keys()):
        v = var_series_dict[H]
        if v.size == 0: 
            continue
        v_um2 = v * (ESCALA_UM_PX**2)
        Hs.append(H)
        means.append(float(np.nanmean(v_um2)))
        # std con NaN-safe; ddof=1 si hay >=2 válidos
        valid = v_um2[np.isfinite(v_um2)]
        stds.append(float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0)
    return np.array(Hs), np.array(means), np.array(stds)

def main():
    fo_files = build_fo_filelist(ROOT_FO)
    if not fo_files:
        print("[ERROR] No se encontraron imágenes FO. Revisá ROOT_FO.")
        return

    # Por campo, computo ambas varianzas (lista por pares)
    series_pk = {}
    series_rad = {}
    for H, files in sorted(fo_files.items()):
        var_pk, var_rad = compute_series_for_field(files)
        series_pk[H]  = var_pk
        series_rad[H] = var_rad

    # Estadísticos por campo (media ± desvío) en µm²
    Hs_pk, m_pk, s_pk = stats_by_field(series_pk)
    Hs_rd, m_rd, s_rd = stats_by_field(series_rad)

    # Plot comparativo
    plt.figure(figsize=(10.5, 5.2))
    if Hs_pk.size:
        plt.errorbar(Hs_pk, m_pk, yerr=s_pk, fmt='o-', lw=1.6, capsize=4,
                     label='Mi método (fajas Pk)', zorder=3)
    if Hs_rd.size:
        plt.errorbar(Hs_rd, m_rd, yerr=s_rd, fmt='s--', lw=1.6, capsize=4,
                     label='Radio efectivo (clase)', zorder=2)

    plt.xlabel('Campo H (Oe)')
    plt.ylabel(r'Dispersión $\mathrm{Var}(u)$ [$\mu$m$^2$]')
    plt.title('Dispersión por campo — comparación de métodos (FO)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # (Opcional) imprimir un resumen numérico
    print("\n=== Resumen por campo (FO) ===")
    print(f"{'H (Oe)':>6} | {'Var_pk (µm²)':>14} | {'Var_rad (µm²)':>14}")
    for H in sorted(set(Hs_pk.tolist()+Hs_rd.tolist())):
        vpk = np.nan
        vr  = np.nan
        if H in series_pk and series_pk[H].size:
            vpk = float(np.nanmean(series_pk[H]*(ESCALA_UM_PX**2)))
        if H in series_rad and series_rad[H].size:
            vr = float(np.nanmean(series_rad[H]*(ESCALA_UM_PX**2)))
        print(f"{H:6d} | {vpk:14.6f} | {vr:14.6f}")

if __name__ == "__main__":
    main()
