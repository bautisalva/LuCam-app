# -*- coding: utf-8 -*-
"""
series_fourier_batch.py
=======================

Procesa una serie de imágenes binarias con patrón de nombre "<prefijo>-<índice><ext>",
parametriza el contorno, selecciona K óptimo por RMS radial, reconstruye la curva
analítica, alinea todas las curvas con una vertical global (x_ref tomada del frame 0),
y guarda TODO en un único NPZ. Además genera dos figuras de diagnóstico opcionales.

---------------------------------------------------------------------------
SALIDAS (en OUTDIR):
  - series_results.npz  ← único archivo con resultados de TODOS los frames
  - combined_analytic_overlay.png  ← analíticas superpuestas (inicio marcado + flecha)
  - combined_rs_stacked.png        ← r(s) apiladas (una debajo de la otra)

NPZ contiene (arrays por frame):
  frames:int[]
  files:object[]                nombres de archivo
  K_curve:int[]                 K óptimo por tolerancia radial
  L:float[]                     perímetro
  x_ref:float                   vertical global (= x_c del frame 0)
  t_al:object[]                 cada elemento es un np.ndarray (s/L) reparametrizado [0,1)
  z_curve_al:object[]           curva analítica COMPLEJA alineada (x+iy)
  r_al:object[]                 perfil radial absoluto alineado
  err_at_K:float[]              RMS radial en K* (diagnóstico)
  start_index:int[]             índice original de arranque antes del roll
  start_point:float[:,2]        puntos (x0,y0) exactos tras alinear
  meta:str                      JSON con parámetros del experimento

---------------------------------------------------------------------------
USO RÁPIDO (por defecto procesa 0..100 en la carpeta actual):
    python series_fourier_batch.py

Para configurar carpeta, prefijo, rango y salida:
    python series_fourier_batch.py \
        --input-dir "E:/tifs" \
        --prefix "Bin-P8139-190Oe-30ms-5Tw" \
        --ext ".tif" \
        --start 0 --end 100 \
        --outdir "out_agg" \
        --M 8192 --eps 1e-3 \
        --assume-binary

Si no querés las figuras, agregá:
        --no-figures
"""

from __future__ import annotations
import os, json, argparse
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# --- Dependencias de imagen ---
from skimage import io, measure
try:
    from skimage.color import rgb2gray
    _HAS_RGB2GRAY = True
except Exception:
    _HAS_RGB2GRAY = False


# =============================================================================
# BLOQUE A — Funciones base (replicadas y simplificadas del pipeline v8)
# =============================================================================

def longest_contour_yx_from_image(path: str, assume_binary: bool = True) -> np.ndarray:
    """
    Lee `path` y devuelve el contorno (y,x) MÁS LARGO (float64).

    Si `assume_binary=True`, la imagen se considera binaria si >0. En caso contrario,
    se normaliza [0,1] y se umbraliza en 0.5. Usa `skimage.measure.find_contours`.

    Returns
    -------
    C : (N,2) float64 en orden (y,x)
    """
    img = io.imread(path)
    if img.ndim == 3:  # RGB → gris
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
        raise RuntimeError(f"No se encontraron contornos en: {path}")
    C = max(cs, key=lambda c: c.shape[0]).astype(np.float64)
    return C


def resample_closed_curve_arclength(P_yx: np.ndarray, M: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Remuestrea una curva cerrada (y,x) a M puntos equiespaciados en longitud de arco.

    Parameters
    ----------
    P_yx : (N,2) float64  curva cerrada en (y,x)
    M    : int            número de muestras uniformes

    Returns
    -------
    P_u : (M,2) float64   curva remuestreada en (y,x)
    s_u : (M,) float64    abscisa curvilínea acumulada en [0, L)
    L   : float           perímetro total de la curva
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


def fourier_descriptors_z(P_yx_u: np.ndarray, remove_translation: bool = True) -> Dict[str, np.ndarray]:
    """
    FFT compleja de z = x + i y en malla s-uniforme; opcionalmente quita traslación.

    Returns
    -------
    dict(Z=FFT(z), z_mean=<complejo> centroide si remove_translation=True)
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
    Reconstrucción truncada usando armónicos ±1..±K_keep (y DC) en la malla original.

    Returns
    -------
    z_rec : (M,) complejo
    """
    Mtot = len(Z)
    K = int(max(1, min(K_keep, Mtot//2)))
    Zt = np.zeros_like(Z)
    Zt[0] = Z[0]
    for k in range(1, K+1):
        Zt[k]   = Z[k]
        Zt[-k]  = Z[-k]
    return np.fft.ifft(Zt) + z_mean


def rms_radial(z_ref: np.ndarray, z_test: np.ndarray) -> float:
    """
    RMS del perfil radial de z_test respecto al centroide de z_ref.
    """
    xr, yr = np.real(z_ref), np.imag(z_ref)
    cx, cy = xr.mean(), yr.mean()
    rr = np.hypot(xr - cx, yr - cy)
    xt, yt = np.real(z_test), np.imag(z_test)
    rt = np.hypot(xt - cx, yt - cy)
    return float(np.sqrt(np.mean((rt - rr)**2)))


def auto_select_K_curve_eps(
    Z: np.ndarray,
    z_u: np.ndarray,
    eps_curve: float = 1e-3,
    Kmin: int = 8,
    refine_steps: int = 8,
    max_candidates: int = 36
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Selección automática de K por tolerancia en RMS radial.

    Retorna
    -------
    K_curve : int
    Ks      : (Q,) ints   valores probados
    errs    : (Q,) floats errores RMS correspondientes
    """
    half = len(Z)//2
    Kmax = max(Kmin+1, half-1)
    Ks = np.unique(np.geomspace(Kmin, Kmax, num=min(max_candidates, Kmax-Kmin+1)).astype(int))
    errs = []
    crossed = False
    last_ok, last_bad = None, None

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

    # Refinamiento bisección
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
    idx = np.argsort(Ks)
    return int(K_curve), Ks[idx], np.array(errs)[idx]


# =============================================================================
# BLOQUE B — Utilidades de alineación y visualización
# =============================================================================

def polygon_signed_area(z: np.ndarray) -> float:
    """Área firmada (shoelace). >0 ⇒ CCW."""
    x = np.real(z); y = np.imag(z)
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

def enforce_ccw(z: np.ndarray) -> np.ndarray:
    """Fuerza sentido CCW en z."""
    return z if polygon_signed_area(z) > 0 else z[::-1].copy()

def compute_rs(z: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """r(s) absoluto respecto de (cx, cy)."""
    x, y = np.real(z), np.imag(z)
    return np.hypot(x - cx, y - cy)

def cross_with_vertical_top(z: np.ndarray, x_ref: float) -> Tuple[int, complex]:
    """
    Cruce EXÁCTO con la vertical global x = x_ref que tenga y MÁXIMO.
    Devuelve:
      - start_idx: índice (j=i+1) para el roll circular
      - z_cross:   punto complejo exacto del cruce
    Fallback: punto con |x - x_ref| mínimo y y máximo.
    """
    x = np.real(z); y = np.imag(z)
    n = len(z)
    best_y, best_i, best_z = None, None, None
    for i in range(n):
        j = (i + 1) % n
        x0, x1 = x[i] - x_ref, x[j] - x_ref
        if x0 == x1:
            continue
        if (x0 <= 0 < x1) or (x1 <= 0 < x0):
            t = -x0 / (x1 - x0)
            y_cross = y[i] + t * (y[j] - y[i])
            if (best_y is None) or (y_cross > best_y):
                best_y = y_cross
                best_i = i
                best_z = x_ref + 1j*y_cross
    if best_i is not None:
        return (best_i + 1) % n, best_z
    # Fallback robusto
    dx = np.abs(x - x_ref)
    idxs = np.argsort(dx)
    eps = max(1e-9, 1e-6 * (np.max(dx) + 1.0))
    cand = idxs[dx[idxs] <= dx[idxs[0]] + eps]
    k = int(cand[np.argmax(y[cand])])
    return k, (x_ref + 1j*y[k])

def roll_with_exact_start(z: np.ndarray, r: np.ndarray, start_idx: int, z_start: complex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Roll circular para que `start_idx` quede al inicio y reemplaza z[0] por el
    cruce exacto `z_start`. Actualiza r[0] usando el centroide post-roll.
    """
    z_rolled = np.roll(z, -start_idx)
    r_rolled = np.roll(r, -start_idx)
    z_rolled[0] = z_start
    cx, cy = np.real(z_rolled).mean(), np.imag(z_rolled).mean()
    r_rolled[0] = np.hypot(np.real(z_start)-cx, np.imag(z_start)-cy)
    return z_rolled, r_rolled

def arrow_from_first_segment(z: np.ndarray, scale: float = 0.06) -> Tuple[float, float]:
    """Vector flecha desde z[0] hacia z[1], escalado al tamaño de la curva."""
    x, y = np.real(z), np.imag(z)
    dx, dy = np.real(z[1]-z[0]), np.imag(z[1]-z[0])
    L = np.hypot(dx, dy) + 1e-12
    dx, dy = dx/L, dy/L
    sx = (x.max()-x.min()) * scale + 1e-12
    sy = (y.max()-y.min()) * scale + 1e-12
    return dx*sx, dy*sy


# =============================================================================
# BLOQUE C — Lógica batch + CLI
# =============================================================================

def find_existing_frames(input_dir: str, prefix: str, ext: str, start: int, end: int) -> List[int]:
    """
    Devuelve la lista de índices [start..end] cuyas rutas existen en disco.
    Permite “agujeros” (los ignora).
    """
    frames = []
    for i in range(start, end+1):
        path = os.path.join(input_dir, f"{prefix}-{i}{ext}")
        if os.path.exists(path):
            frames.append(i)
    if not frames:
        raise FileNotFoundError("No se encontró ninguna imagen con el patrón solicitado.")
    return frames


def process_series(
    input_dir: str,
    prefix: str,
    ext: str,
    frames: List[int],
    outdir: str,
    M: int,
    eps: float,
    assume_binary: bool,
    make_figures: bool,
    plot_max_frames: int = 200
):
    """
    Procesamiento batch completo:
      1) Usa frame 0 (primero de `frames`) para fijar x_ref.
      2) Procesa todos los frames con esa vertical global x_ref.
      3) Guarda NPZ único y figuras opcionales.
    """
    os.makedirs(outdir, exist_ok=True)

    # ---------- Primer pasada: frame de referencia (frames[0]) ----------
    idx0 = frames[0]
    f0 = os.path.join(input_dir, f"{prefix}-{idx0}{ext}")
    C0 = longest_contour_yx_from_image(f0, assume_binary=assume_binary)
    P0, s0, L0 = resample_closed_curve_arclength(C0, M=M)
    z0_u = P0[:,1] + 1j*P0[:,0]
    fd0 = fourier_descriptors_z(P0, remove_translation=True)
    Z0, z0_mean = fd0["Z"], fd0["z_mean"]
    K0, Ks0, errs0 = auto_select_K_curve_eps(Z0, z_u=z0_u, eps_curve=eps)
    z0_curve = reconstruct_from_Z(Z0, K0, z_mean=z0_mean)
    z0_curve = enforce_ccw(z0_curve)
    x_ref = float(np.real(z0_curve).mean())  # vertical global

    # ---------- Contenedores ----------
    all_frames, all_files = [], []
    all_K, all_L, all_errK = [], [], []
    all_t_al, all_z_al, all_r_al = [], [], []
    all_start, all_p0 = [], []

    # ---------- Procesamiento por frame ----------
    for idx in frames:
        fpath = os.path.join(input_dir, f"{prefix}-{idx}{ext}")
        print(f"[INFO] Frame {idx} → {os.path.basename(fpath)}")

        # Contorno y remuestreo
        C = longest_contour_yx_from_image(fpath, assume_binary=assume_binary)
        P, s_u, L = resample_closed_curve_arclength(C, M=M)
        z_u = P[:,1] + 1j*P[:,0]

        # FFT → K óptimo → reconstrucción
        fd = fourier_descriptors_z(P, remove_translation=True)
        Z, z_mean = fd["Z"], fd["z_mean"]
        K_curve, Ks_try, errs_try = auto_select_K_curve_eps(Z, z_u=z_u, eps_curve=eps)
        z_curve = reconstruct_from_Z(Z, K_curve, z_mean=z_mean)
        z_curve = enforce_ccw(z_curve)

        # r(s) respecto a su centroide
        xc, yc = np.real(z_curve).mean(), np.imag(z_curve).mean()
        r_ana = compute_rs(z_curve, xc, yc)

        # Alineación por cruce exacto con x_ref (y máximo)
        start_idx, z_start = cross_with_vertical_top(z_curve, x_ref)
        z_al, r_al = roll_with_exact_start(z_curve, r_ana, start_idx, z_start)
        t_al = np.linspace(0.0, 1.0, len(z_al), endpoint=False)

        # Error en K* (diagnóstico)
        kpos = int(np.argmin(np.abs(Ks_try - K_curve)))
        err_at_K = float(errs_try[kpos])

        # Acumular
        all_frames.append(idx)
        all_files.append(fpath)
        all_K.append(int(K_curve))
        all_L.append(float(L))
        all_errK.append(err_at_K)
        all_t_al.append(t_al.astype(np.float64))
        all_z_al.append(z_al.astype(np.complex128))
        all_r_al.append(r_al.astype(np.float64))
        all_start.append(int(start_idx))
        all_p0.append((float(np.real(z_al[0])), float(np.imag(z_al[0]))))

    # ---------- Figuras opcionales ----------
    if make_figures:
        # 1) Analíticas superpuestas
        n_show = min(len(all_frames), plot_max_frames)
        plt.figure(figsize=(7.0, 6.6))
        plt.axvline(x_ref, color="k", lw=0.8, alpha=0.3, label=f"x_ref={x_ref:.3f}")
        for i in range(n_show):
            idx, zc, Kc, p0 = all_frames[i], all_z_al[i], all_K[i], all_p0[i]
            plt.plot(np.real(zc), np.imag(zc), lw=1.1, label=f"f{idx} (K*={Kc})")
            plt.plot(p0[0], p0[1], "ko", ms=3)
            dx, dy = arrow_from_first_segment(zc, scale=0.06)
            plt.arrow(p0[0], p0[1], dx, dy, head_width=0.02*max(abs(dx), abs(dy), 1.0),
                      length_includes_head=True, color="k", lw=0.9)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Curvas analíticas alineadas (vertical global x = x_ref)")
        plt.xlabel("x [px]"); plt.ylabel("y [px]")
        plt.legend(frameon=False, fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "combined_analytic_overlay.png"), dpi=220)
        plt.close()

        # 2) r(s) apiladas
        n = len(all_frames)
        fig, axes = plt.subplots(n, 1, figsize=(7.6, 2.0*n), constrained_layout=True, sharex=True)
        if n == 1: axes = [axes]
        for ax, idx, t_i, r_i in zip(axes, all_frames, all_t_al, all_r_al):
            ax.plot(t_i, r_i, lw=1.0)
            ax.set_ylabel("r(s) [px]")
            ax.set_title(f"frame {idx}", fontsize=10)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("s/L")
        fig.suptitle("Perfiles radiales r(s) — alineados por x_ref y apilados", y=1.02, fontsize=12)
        fig.savefig(os.path.join(outdir, "combined_rs_stacked.png"), dpi=220)
        plt.close(fig)

    # ---------- Guardado único ----------
    meta = dict(
        input_dir=input_dir, prefix=prefix, ext=ext, frames=frames,
        outdir=outdir, M=M, eps=eps, assume_binary=assume_binary,
        x_ref=x_ref
    )
    np.savez(
        os.path.join(outdir, "series_results.npz"),
        frames=np.array(all_frames, dtype=int),
        files=np.array(all_files, dtype=object),
        K_curve=np.array(all_K, dtype=int),
        L=np.array(all_L, dtype=float),
        x_ref=np.array([x_ref], dtype=float),
        t_al=np.array(all_t_al, dtype=object),
        z_curve_al=np.array(all_z_al, dtype=object),
        r_al=np.array(all_r_al, dtype=object),
        err_at_K=np.array(all_errK, dtype=float),
        start_index=np.array(all_start, dtype=int),
        start_point=np.array(all_p0, dtype=float),
        meta=json.dumps(meta, ensure_ascii=False)
    )
    print(f"\n[OK] Guardado único → {os.path.join(outdir, 'series_results.npz')}")
    if make_figures:
        print("[OK] Figuras → combined_analytic_overlay.png, combined_rs_stacked.png")


def parse_args():
    p = argparse.ArgumentParser(description="Batch de parametrización Fourier con alineación global.")
    p.add_argument("--input-dir", type=str, default=".", help="Carpeta con las imágenes.")
    p.add_argument("--prefix", type=str, default="Bin-P8139-190Oe-30ms-5Tw", help="Prefijo común del nombre.")
    p.add_argument("--ext", type=str, default=".tif", help="Extensión (incluyendo punto).")
    p.add_argument("--start", type=int, default=0, help="Índice inicial (inclusive).")
    p.add_argument("--end", type=int, default=2, help="Índice final (inclusive).")
    p.add_argument("--outdir", type=str, default="out_pablo", help="Carpeta de salida.")
    p.add_argument("--M", type=int, default=8192, help="Muestras uniformes sobre la curva.")
    p.add_argument("--eps", type=float, default=1e-3, help="Tolerancia para seleccionar K.")
    p.add_argument("--assume-binary", action="store_true", help="Tratar las imágenes como binarias (img>0).")
    p.add_argument("--no-figures", action="store_true", help="No generar figuras.")
    p.add_argument("--plot-max-frames", type=int, default=200, help="Máximo de curvas a dibujar en el overlay.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frames = find_existing_frames(args.input_dir, args.prefix, args.ext, args.start, args.end)
    process_series(
        input_dir=args.input_dir,
        prefix=args.prefix,
        ext=args.ext,
        frames=frames,
        outdir=args.outdir,
        M=args.M,
        eps=args.eps,
        assume_binary=args.assume_binary,
        make_figures=not args.no_figures,
        plot_max_frames=args.plot_max_frames
    )
