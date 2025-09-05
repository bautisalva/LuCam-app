# -*- coding: utf-8 -*-
"""
domains_v8_radial_only.py
=========================
Versión v8 — Método radial absoluto r(s), limpio y orientado a figuras de paper.
Se trabaja sobre una imagen **binarizada** (0/255 o 0/1), sin reducción.

Figuras generadas
-----------------
01_overlay_modes.png : Overlay del contorno remuestreado + varios modos bajos (K en overlay_modes)
                       + curva final (K=K_curve). La curva final se colorea por s/L para indicar el recorrido.
02_error_vs_K.png    : Curva de error RMS radial vs armónico K (muestreo denso controlado).
03_xy_vs_s.png       : Doble plot x(s) e y(s) (original y curva final).
04_radial_rs.png     : r(s) absoluto de la curva final (respecto del centroide).

Salida numérica
---------------
results_v8.npz con: C, P_u, s_u, L, Z, z_mean, K_curve, Ks_err, errs, z_curve, t, r

Uso rápido
----------
from domains_v8_radial_only import run_v8

run_v8(
    image_path=r"E:\Documents\Labo 6\LuCam-app\analisis\parametrizacion\Bin-P8139-190Oe-30ms-5Tw-99.tif",
    outdir="v8_out",
    M=8192,
    eps_curve=1e-3,                 # tolerancia para elegir K_curve (menor => más K)
    overlay_modes=(1,2,4,8,16,32),  # modos de baja frecuencia a mostrar en el overlay
    assume_binary=True
)
"""
from __future__ import annotations

import os
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from skimage import io, measure

try:
    from skimage.color import rgb2gray
    _HAS_RGB2GRAY = True
except Exception:
    _HAS_RGB2GRAY = False


# -----------------------------------------------------------------------------
# 1) Contorno desde imagen binaria (rápido)
# -----------------------------------------------------------------------------
def longest_contour_yx_from_image(path: str, assume_binary: bool = True):
    """
    Lee la imagen y devuelve el contorno (y,x) más largo.
    Si assume_binary=True, define binario = (img>0) y va directo a find_contours.
    """
    img = io.imread(path)
    if img.ndim == 3:
        if _HAS_RGB2GRAY:
            img = rgb2gray(img)
        else:
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


# -----------------------------------------------------------------------------
# 2) Remuestreo por longitud de arco (curva cerrada → M muestras uniformes)
# -----------------------------------------------------------------------------
def resample_closed_curve_arclength(P_yx: np.ndarray, M: int):
    """
    P_yx: (N,2) en (y,x). Devuelve:
      P_u: (M,2) uniforme en longitud de arco (y,x)
      s_u: (M,) curvilínea acumulada en [0,L)
      L  : perímetro
    """
    P = np.asarray(P_yx, float)
    d = np.diff(np.vstack([P, P[:1]]), axis=0)
    seg = np.hypot(d[:,0], d[:,1])
    s = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    L = s[-1] + seg[-1]
    s_u = np.linspace(0.0, L, M, endpoint=False)
    y_u = np.interp(s_u, s, P[:,0])
    x_u = np.interp(s_u, s, P[:,1])
    return np.column_stack([y_u, x_u]), s_u, L


# -----------------------------------------------------------------------------
# 3) Fourier complejo sobre s
# -----------------------------------------------------------------------------
def fourier_descriptors_z(P_yx_u: np.ndarray, remove_translation: bool = True) -> Dict[str, np.ndarray]:
    """
    P_yx_u: (M,2) s-uniforme. Devuelve Z (FFT de z) y z_mean (si remove_translation=True).
    """
    y = P_yx_u[:,0]; x = P_yx_u[:,1]
    z = x + 1j*y
    z_mean = z.mean()
    if remove_translation:
        z = z - z_mean
    Z = np.fft.fft(z)
    return dict(Z=Z, z_mean=z_mean)


def reconstruct_from_Z(Z: np.ndarray, K_keep: int, z_mean: complex = 0.0) -> np.ndarray:
    """
    Reconstrucción truncada con K_keep armónicos positivos (y negativos simétricos).
    Devuelve z_rec (M,) en la misma malla s.
    """
    Mtot = len(Z)
    K = int(max(1, min(K_keep, Mtot//2)))
    Zt = np.zeros_like(Z)
    Zt[0] = Z[0]
    for k in range(1, K+1):
        Zt[k] = Z[k]
        Zt[-k] = Z[-k]
    return np.fft.ifft(Zt) + z_mean


# -----------------------------------------------------------------------------
# 4) Selección automática *simple* de K_curve (por tolerancia de RMS radial)
# -----------------------------------------------------------------------------
def rms_radial(z_ref: np.ndarray, z_test: np.ndarray) -> float:
    """RMS del perfil radial respecto al centroide del ref."""
    xr, yr = np.real(z_ref), np.imag(z_ref)
    cx, cy = xr.mean(), yr.mean()
    rr = np.hypot(xr - cx, yr - cy)
    xt, yt = np.real(z_test), np.imag(z_test)
    rt = np.hypot(xt - cx, yt - cy)
    return float(np.sqrt(np.mean((rt - rr)**2)))

def auto_select_K_curve_eps(Z: np.ndarray, z_u: np.ndarray, eps_curve: float = 1e-3,
                            Kmin: int = 8, refine_steps: int = 8, max_candidates: int = 36):
    """
    Devuelve K_curve, Ks_probadas, errs(K).
    Estrategia:
      1) Barrido geométrico entre Kmin y Nyquist (≈M/2) con ~max_candidates valores.
      2) Si encuentra cruce (err <= eps), refina con búsqueda binaria entre el anterior y el actual.
      3) Si no cruza, devuelve Nyquist-1.
    """
    half = len(Z)//2
    Kmax = max(Kmin+1, half-1)
    Ks = np.unique(np.geomspace(Kmin, Kmax, num=min(max_candidates, Kmax-Kmin+1)).astype(int))
    errs = []
    crossed = False
    last_ok = None
    last_bad = None

    for K in Ks:
        zr = reconstruct_from_Z(Z, K, z_mean=0.0)  # z_u ya contiene z_mean
        e = rms_radial(z_u, zr)
        errs.append(e)
        if e <= eps_curve:
            crossed = True
            last_ok = K
            break
        last_bad = K

    if not crossed:
        return int(Ks[-1]), Ks, np.array(errs, float)

    # Refino por bisección entre last_bad y last_ok
    a = last_bad if last_bad is not None else Kmin
    b = last_ok
    for _ in range(refine_steps):
        if b - a <= 1:
            break
        mid = (a + b)//2
        zr = reconstruct_from_Z(Z, mid, z_mean=0.0)
        e = rms_radial(z_u, zr)
        Ks = np.append(Ks, mid)
        errs = np.append(errs, e)
        if e <= eps_curve:
            b = mid
        else:
            a = mid
    K_curve = b
    # Ordenar para graficar
    idx = np.argsort(Ks)
    return int(K_curve), Ks[idx], np.array(errs)[idx]


# -----------------------------------------------------------------------------
# 5) Auxiliar: línea coloreada por parámetro (para la curva final)
# -----------------------------------------------------------------------------
def linecollection_colored_by_param(x: np.ndarray, y: np.ndarray, c: np.ndarray,
                                    cmap: str = "viridis", lw: float = 2.0) -> LineCollection:
    """
    Devuelve un LineCollection para (x,y) con color por c (mismo tamaño) usando segmentos.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=Normalize(0, 1))
    lc.set_array(c)
    lc.set_linewidth(lw)
    return lc


# -----------------------------------------------------------------------------
# 6) Pipeline principal v8 (radial-only)
# -----------------------------------------------------------------------------
def run_v8(image_path: str,
           outdir: str = "v8_out",
           M: int = 8192,
           eps_curve: float = 1e-3,
           overlay_modes: tuple = (1, 2, 4, 8, 16, 32),
           assume_binary: bool = True) -> Dict[str, np.ndarray]:

    os.makedirs(outdir, exist_ok=True)

    # 1) Contorno y remuestreo
    C = longest_contour_yx_from_image(image_path, assume_binary=assume_binary)
    P_u, s_u, L = resample_closed_curve_arclength(C, M=M)  # (y,x)
    z_u = P_u[:,1] + 1j*P_u[:,0]
    t = s_u / L  # s normalizado en [0,1)

    # 2) FFT y selección automática de K_curve
    fd = fourier_descriptors_z(P_u, remove_translation=True)
    Z, z_mean = fd["Z"], fd["z_mean"]
    K_curve, Ks, errs = auto_select_K_curve_eps(Z, z_u=z_u, eps_curve=eps_curve)
    z_curve = reconstruct_from_Z(Z, K_curve, z_mean=z_mean)


    # 3) Figura 01 — Overlay con primeros modos y curva final (final coloreada por s/L)
    fig, ax = plt.subplots(figsize=(6.9, 6.6))
    
    # Modos bajos (solo si K < K_final, colores grises ordenados)
    colors = plt.cm.Greys(np.linspace(0.35, 0.85, len(overlay_modes)))
    for K, col in zip(overlay_modes, colors):
        if K < K_curve:
            zr = reconstruct_from_Z(Z, K, z_mean=z_mean)
            ax.plot(np.real(zr), np.imag(zr), color=col, lw=1.1, label=f"K={K}", zorder=2)
    
    # Curva final con gradiente por s/L
    lc = linecollection_colored_by_param(np.real(z_curve), np.imag(z_curve), t, cmap="viridis", lw=2.0)
    lc.set_zorder(3)
    ax.add_collection(lc)
    
    # Punto inicial
    ax.plot(np.real(z_curve)[0], np.imag(z_curve)[0], 'o', ms=4, color='black', label="Inicio (t=0)", zorder=4)
    # Centroide de la curva final
    xc, yc = np.real(z_curve).mean(), np.imag(z_curve).mean()
    #ax.plot(xc, yc, '.', ms=7, color='red', label="Centroide", zorder=5)
    # Ejes y estética
    ax.set_aspect('equal')
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    
    # Leyenda: agregamos un "proxy" para la curva final (porque LineCollection no entra solo)
    final_proxy = Line2D([0], [0], color='k', lw=2.0)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(final_proxy); labels.append(f"Curva final (K={K_curve})")
    ax.legend(handles, labels, loc='best', fontsize=8, frameon=False)
    
    # Barra de color (recorrido paramétrico s/L)
    cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('s/L')
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_overlay_modes.png"), dpi=220)

    # 4) Figura 02 — Error RMS radial vs K (muestreo denso hasta K_final)
    # Para evitar costos, si K_final>400, muestreamos 400 puntos entre 1 y K_final
    if K_curve <= 400:
        Ks_err = np.arange(1, K_curve+1, dtype=int)
    else:
        Ks_err = np.unique(np.linspace(1, K_curve, 400, dtype=int))
    errs_dense = []
    for K in Ks_err:
        zr = reconstruct_from_Z(Z, K, z_mean=0.0)  # medir vs z_u (que ya incluye z_mean)
        errs_dense.append(rms_radial(z_u, zr))
    errs_dense = np.array(errs_dense, float)

    plt.figure(figsize=(6.6, 4.0))
    plt.plot(Ks_err, errs_dense, '-', lw=1.5, color='#444444')
    plt.axvline(K_curve, ls="--", color='#888888')
    plt.text(K_curve, np.max(errs_dense)*0.93, f"K_final={K_curve}", rotation=90, va="top", ha="right", fontsize=8)
    plt.xlabel('Armónico K'); plt.ylabel('RMS radial (vs contorno)')
    plt.title('Error de reconstrucción vs K')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "02_error_vs_K.png"), dpi=220)

    # 5) Figura 03 — x(s) e y(s) (original y final)
    xr_u, yr_u = np.real(z_u), np.imag(z_u)
    xr_c, yr_c = np.real(z_curve), np.imag(z_curve)
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    ax1.plot(t, xr_u, color='#777777', lw=0.9, label='x(s) original')
    ax1.plot(t, xr_c, color='#1f77b4', lw=1.2, label=f'x(s) K={K_curve}')
    ax1.set_xlabel('s/L'); ax1.set_ylabel('x [px]'); ax1.set_title('x(s)')
    ax1.grid(alpha=0.25); ax1.legend(fontsize=8, frameon=False)
    ax2.plot(t, yr_u, color='#777777', lw=0.9, label='y(s) original')
    ax2.plot(t, yr_c, color='#d62728', lw=1.2, label=f'y(s) K={K_curve}')
    ax2.set_xlabel('s/L'); ax2.set_ylabel('y [px]'); ax2.set_title('y(s)')
    ax2.grid(alpha=0.25); ax2.legend(fontsize=8, frameon=False)
    fig3.tight_layout(); fig3.savefig(os.path.join(outdir, "03_xy_vs_s.png"), dpi=220)

    # 6) Figura 04 — r(s) absoluto de la curva final
    xc, yc = xr_c.mean(), yr_c.mean()
    r = np.hypot(xr_c - xc, yr_c - yc)
    plt.figure(figsize=(7.2, 3.8))
    plt.plot(t, r, '-', lw=1.2, color='#2ca02c')
    plt.xlabel('s/L'); plt.ylabel('r(s) [px]')
    plt.title('Perfil radial absoluto r(s) de la curva final')
    plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "04_radial_rs.png"), dpi=220)

    # 7) Guardado
    np.savez(os.path.join(outdir, "results_v8.npz"),
             C=C, P_u=P_u, s_u=s_u, L=L,
             Z=Z, z_mean=z_mean,
             K_curve=np.array([K_curve]),
             Ks_err=Ks_err, errs=errs_dense,
             z_curve=z_curve, t=t, r=r)

    print(f"[OK] M={M} | K_curve={K_curve} | outdir={outdir}")
    return dict(K_curve=K_curve)


# -----------------------------------------------------------------------------
# Ejecutable
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_v8(
        image_path=r"E:\Documents\Labo 6\LuCam-app\analisis\parametrizacion\Bin-P8139-190Oe-30ms-5Tw-99.tif",
        outdir="v8_out",
        M=8192,
        eps_curve=1e-3,
        overlay_modes=(1,8),
        assume_binary=True
    )
