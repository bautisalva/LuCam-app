# -*- coding: utf-8 -*-
"""
01A_overlay_point2point.py
--------------------------
Genera SOLO la figura "01A_overlay_point2point.png" con:
- Segmentos punto-a-punto S_K[m] -> S_{K+1}[m] para K=K_start..K_end
- Curvas completas S_{K_start} (por defecto 16) y S_{K_end} con marcas de inicio (t=0) y flecha de recorrido.

Convenciones:
- Contorno crudo en (y,x) por skimage.
- Fourier en z = x + i y sobre M puntos s-uniformes (longitud de arco).
- Reconstrucciones S_K están en la MISMA malla de M puntos → correspondencia 1–a–1 por índice m.

Requisitos: numpy, matplotlib, scikit-image
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure

try:
    from skimage.color import rgb2gray
    _HAS_RGB2GRAY = True
except Exception:
    _HAS_RGB2GRAY = False


# ==============================
# Utilidades de contorno / remuestreo
# ==============================
def longest_contour_yx_from_image(path: str, assume_binary: bool = True):
    """
    Lee la imagen y devuelve el contorno (y,x) más largo (array (N,2)).
    Si assume_binary=True, toma binario = (img>0) directamente.
    """
    img = io.imread(path)
    if img.ndim == 3:
        if _HAS_RGB2GRAY:
            img = rgb2gray(img)
        else:
            # luminancia simple con normalización
            img = (0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]).astype(np.float32)
            vmin, vmax = float(img.min()), float(img.max())
            if vmax > vmin:
                img = (img - vmin) / (vmax - vmin)

    if assume_binary:
        binm = img > 0
    else:
        vmin, vmax = float(img.min()), float(img.max())
        img01 = (img - vmin) / (vmax - vmin + 1e-12)
        binm = img01 >= 0.5

    cs = measure.find_contours(binm.astype(float), level=0.5)
    if not cs:
        raise RuntimeError("No se encontraron contornos en la imagen.")
    return max(cs, key=lambda c: c.shape[0])  # (N,2) en (y,x)


def resample_closed_curve_arclength(P_yx: np.ndarray, M: int):
    """
    Remuestrea curva cerrada (y,x) a M puntos uniformes en longitud de arco.
    Devuelve: P_u (M,2 en y,x), s_u (M,), L (perímetro).
    """
    P = np.asarray(P_yx, float)
    # diferenciar con cierre
    d = np.diff(np.vstack([P, P[:1]]), axis=0)
    seg = np.hypot(d[:,0], d[:,1])
    s = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    L = s[-1] + seg[-1]
    s_u = np.linspace(0.0, L, M, endpoint=False)
    y_u = np.interp(s_u, s, P[:,0])
    x_u = np.interp(s_u, s, P[:,1])
    return np.column_stack([y_u, x_u]), s_u, L


# ==============================
# Fourier y reconstrucciones
# ==============================
def fourier_descriptors_z(P_yx_u: np.ndarray, remove_translation: bool = True):
    """
    z = x + i y sobre los M puntos s-uniformes. Devuelve Z (FFT de z) y z_mean.
    """
    y = P_yx_u[:,0]; x = P_yx_u[:,1]
    z = x + 1j*y
    z_mean = z.mean()
    if remove_translation:
        z = z - z_mean
    Z = np.fft.fft(z)
    return dict(Z=Z, z_mean=z_mean)


def reconstruct_from_Z(Z: np.ndarray, K_keep: int, z_mean: complex = 0.0):
    """
    S_K: reconstrucción con armónicos ±1..±K_keep (más el término 0).
    Devuelve z_rec (M,) complejo (x+iy).
    """
    Mtot = len(Z)
    K = int(max(1, min(K_keep, Mtot//2)))
    Zt = np.zeros_like(Z)
    Zt[0] = Z[0]
    for k in range(1, K+1):
        Zt[k] = Z[k]
        Zt[-k] = Z[-k]
    return np.fft.ifft(Zt) + z_mean


def auto_select_K_curve_eps(Z: np.ndarray, z_u: np.ndarray, eps_curve: float = 1e-3,
                            Kmin: int = 8, refine_steps: int = 8, max_candidates: int = 36):
    """
    Selección automática de K_curve por tolerancia de RMS radial.
    Devuelve K_curve (int), Ks probadas (array), errs (array).
    """
    def rms_radial(z_ref, z_test):
        xr, yr = np.real(z_ref), np.imag(z_ref)
        cx, cy = xr.mean(), yr.mean()
        rr = np.hypot(xr - cx, yr - cy)
        xt, yt = np.real(z_test), np.imag(z_test)
        rt = np.hypot(xt - cx, yt - cy)
        return float(np.sqrt(np.mean((rt - rr)**2)))

    half = len(Z)//2
    Kmax = max(Kmin+1, half-1)
    Ks = np.unique(np.geomspace(Kmin, Kmax, num=min(max_candidates, Kmax-Kmin+1)).astype(int))
    errs, crossed, last_ok, last_bad = [], False, None, None

    for K in Ks:
        zr = reconstruct_from_Z(Z, K, z_mean=0.0)
        e = rms_radial(z_u, zr); errs.append(e)
        if e <= eps_curve:
            crossed = True; last_ok = K; break
        last_bad = K

    if not crossed:
        return int(Ks[-1]), Ks, np.array(errs, float)

    a = last_bad if last_bad is not None else Kmin
    b = last_ok
    for _ in range(refine_steps):
        if b - a <= 1: break
        mid = (a + b)//2
        zr = reconstruct_from_Z(Z, mid, z_mean=0.0)
        e = rms_radial(z_u, zr)
        Ks = np.append(Ks, mid)
        errs = np.append(errs, e)
        if e <= eps_curve: b = mid
        else: a = mid

    K_curve = b
    idx = np.argsort(Ks)
    return int(K_curve), Ks[idx], np.array(errs)[idx]


def reconstruct_many(Z: np.ndarray, Ks: list[int], z_mean: complex = 0.0):
    """Lista [S_K] para K en Ks (cada S_K es (M,) complejo)."""
    return [reconstruct_from_Z(Z, int(K), z_mean=z_mean) for K in Ks]


# ==============================
# Segmentos entre curvas consecutivas
# ==============================
def segments_between_many(z_list: list[np.ndarray]) -> np.ndarray:
    """
    Genera segmentos S_K[m] -> S_{K+1}[m] para cada m, a partir de z_list=[S_K,...,S_Kend].
    Devuelve array (num_segments, 2, 2) con pares (x,y).
    """
    assert len(z_list) >= 2
    Mloc = len(z_list[0])
    segs = []
    for m in range(Mloc):
        # cadena S_K[m] -> S_{K+1}[m] -> ...
        chain = np.array([[np.real(zi[m]), np.imag(zi[m])] for zi in z_list], float)
        for a, b in zip(chain[:-1], chain[1:]):
            segs.append([a, b])
    return np.array(segs)


# ==============================
# Figura 01A: overlay punto-a-punto (K_start..K_end)
# ==============================
def figure_overlay_point2point(image_path: str,
                               outdir: str = "out_overlay_pp",
                               M: int = 4096,
                               eps_curve: float = 1e-3,
                               assume_binary: bool = True,
                               K_start: int = 16,
                               K_end: int | None = None,
                               stride_overlay: int = 8):
    """
    Produce "01A_overlay_point2point.png" en outdir.
    - K_start..K_end: rango de armónicos (si K_end=None -> usa K_curve).
    - stride_overlay: submuestreo de puntos (para no saturar visualmente).
    """
    os.makedirs(outdir, exist_ok=True)

    # 1) Contorno y remuestreo
    C = longest_contour_yx_from_image(image_path, assume_binary=assume_binary)
    P_u, s_u, L = resample_closed_curve_arclength(C, M=M)   # (y,x)
    z_u = P_u[:,1] + 1j*P_u[:,0]

    # 2) FFT y K_curve
    fd = fourier_descriptors_z(P_u, remove_translation=True)
    Z, z_mean = fd["Z"], fd["z_mean"]
    K_curve, _, _ = auto_select_K_curve_eps(Z, z_u=z_u, eps_curve=eps_curve)

    # 3) Rango K
    half = M//2
    K_first = int(max(1, K_start))
    K_last  = int(K_curve if K_end is None else min(K_end, half-1))
    if K_last < K_first:
        K_last = K_first
    Ks_pp_full = list(range(K_first, K_last+1))

    # 4) Reconstrucciones y (opcional) submuestreo para ploteo de segmentos
    zK_pp_full = reconstruct_many(Z, Ks_pp_full, z_mean=z_mean)  # lista de S_K
    if stride_overlay > 1:
        zK_pp_draw = [zk[::stride_overlay] for zk in zK_pp_full]
    else:
        zK_pp_draw = zK_pp_full

    # 5) Segmentos S_K[m] -> S_{K+1}[m]
    segs = segments_between_many(zK_pp_draw)  # (Nseg, 2, 2)

    # 6) Figura
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    plt.figure(figsize=(7.2, 6.9))
    ax = plt.gca()

    # Curva S_{K_first}: dibujar completa + inicio + flecha
    z_first = zK_pp_full[0]
    ax.plot(np.real(z_first), np.imag(z_first), lw=1.2, color='#1f77b4', label=f"S_{K_first}")
    ax.plot(np.real(z_first)[0], np.imag(z_first)[0], 'o', ms=5, color='#1f77b4')
    ax.annotate("",
                xy=(np.real(z_first)[1], np.imag(z_first)[1]),
                xytext=(np.real(z_first)[0], np.imag(z_first)[0]),
                arrowprops=dict(arrowstyle="->", lw=1.0, color='#1f77b4'))

    # Curva S_{K_last}: completa + inicio + flecha
    z_last = zK_pp_full[-1]
    ax.plot(np.real(z_last), np.imag(z_last), lw=1.2, color='#d62728', label=f"S_{K_last}")
    ax.plot(np.real(z_last)[0], np.imag(z_last)[0], 'o', ms=5, color='#d62728')
    ax.annotate("",
                xy=(np.real(z_last)[1], np.imag(z_last)[1]),
                xytext=(np.real(z_last)[0], np.imag(z_last)[0]),
                arrowprops=dict(arrowstyle="->", lw=1.0, color='#d62728'))

    # Nube de puntos tenue para intermedios (downsampleados)
    cmap_pts = plt.cm.plasma
    n_lists = len(zK_pp_draw)
    for i, zk in enumerate(zK_pp_draw):
        col = cmap_pts(0.15 + 0.7 * (i / max(1, n_lists - 1)))
        ax.scatter(np.real(zk), np.imag(zk), s=6, color=col, alpha=0.7,
                   label=f"S_{Ks_pp_full[0] + i}" if i in (0, n_lists-1) else None, zorder=3)

    # Segmentos entre S_K y S_{K+1}
    lc_pp = LineCollection(segs, colors='#444444', linewidths=0.5, alpha=0.6)
    ax.add_collection(lc_pp)

    ax.set_aspect('equal')
    ax.set_xlabel('x [px]'); ax.set_ylabel('y [px]')
    ax.set_title(f"Overlay punto-a-punto: K={K_first}…{K_last} (segmentos S_K[m]→S_{{K+1}}[m])")

    # Leyenda compacta asegurando extremos
    handles, labels = ax.get_legend_handles_labels()
    want = {f"S_{K_first}": '#1f77b4', f"S_{K_last}": '#d62728'}
    for name, col in want.items():
        if name not in labels:
            handles.append(plt.Line2D([0],[0], color=col, lw=1.5))
            labels.append(name)
    ax.legend(handles, labels, loc='best', fontsize=8, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(outdir, "01A_overlay_point2point.png")
    plt.savefig(out_path, dpi=220)
    print(f"[OK] Figura guardada en: {out_path}")


# ==============================
# Ejecutable
# ==============================
if __name__ == "__main__":
    # EDITÁ ESTOS PARÁMETROS:
    IMAGE_PATH = r"C:\Users\Marina\Documents\Labo 6\LuCam-app\analisis\parametrizacion\Bin-P8139-190Oe-30ms-5Tw-99.tif"
    OUTDIR     = "out_overlay_pp"
    M          = 80
    EPS_CURVE  = 1e-3
    ASSUME_BIN = True
    K_START    = 16
    K_END      = None      # None => usa K_curve; o poné un entero <= M//2 - 1
    STRIDE     = 8         # aumentar (12,16,...) si el gráfico queda muy cargado

    figure_overlay_point2point(
        image_path=IMAGE_PATH,
        outdir=OUTDIR,
        M=M,
        eps_curve=EPS_CURVE,
        assume_binary=ASSUME_BIN,
        K_start=K_START,
        K_end=K_END,
        stride_overlay=STRIDE
    )
