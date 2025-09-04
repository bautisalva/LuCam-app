
"""
EFD v5 (EFD-only, rápido) — spatial_efd NO normalizado + baseline auto + opción imagen binaria
==============================================================================================

Resumen
-------
- **Modo EFD-only**: calculamos EFD una sola vez hasta Nyquist y reconstruimos sobre la
  MISMA malla paramétrica t dos curvas:
    * Baseline de baja frecuencia (H_base)
    * Curva final de alta frecuencia (H_curve)
  Aplanado: h(t) = (r_curve(t) - r_base(t)) · N_base(t), donde N_base es la normal unitaria
  de la baseline. No se reparametriza el contorno *raw*.

- **Selección automática de armónicos**:
    * H_curve por potencia acumulada (umbral p_curve).
    * H_base por potencia acumulada (p_base) o por un criterio de trade-off (opcional).

- **Entrada por imagen binaria** (rápida): si tu .tif *ya es binario*, saltamos Otsu y
  morfología y vamos directo a `find_contours`. No se reduce resolución.

Dependencias
------------
pip install spatial_efd numpy matplotlib scikit-image

Salida
------
- 01_overlay.png                 : overlay (Original/contorno + Baseline + Curva)
- 02_error_vs_H.png              : RMS vs H (con líneas H_base, H_curve)
- 03_efd_spectrum.png            : amplitud EFD por armónico
- 04_cumulative_power.png        : potencia acumulada
- 05_unroll_h.png                : h(t)
- 06_roughness_fft.png           : FFT de h(t)
- results_v5_fast.npz            : arrays útiles (coef, Hs, t, h, etc.)

Uso rápido
----------
from efd_v5_efd_only_fast import demo_v5

# Sintético
demo_v5(mode="synthetic", outdir="efd_v5_out", M=8192, p_curve=0.999, p_base=0.95)

# Imagen binaria (sin reducir resolución)
demo_v5(mode="image",
        image_path=r"C:\ruta\a\tu.tif",
        assume_binary=True,
        outdir="efd_v5_out",
        M=8192, p_curve=0.999, p_base=0.95)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Optional, Iterable

import numpy as np
import matplotlib.pyplot as plt

import spatial_efd as sefd

# Módulos mínimos de scikit-image
from skimage import io, measure
try:
    from skimage.color import rgb2gray
    _HAS_RGB2GRAY = True
except Exception:
    _HAS_RGB2GRAY = False


# -----------------------------------------------------------------------------
# Utilidades de curva (trabajamos con curvas *abiertas*; se cierran sólo al plotear)
# -----------------------------------------------------------------------------
def close_curve(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve la curva *cerrada* repitiendo el primer punto al final (para plotting)."""
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    return x, y

def canonicalize_open(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Origen canónico estable para curvas CERRADAS: índice del mínimo x (desempate por y).
    Aquí: cerramos temporalmente, rotamos y devolvemos *abierta* (sin duplicar cierre).
    """
    xc, yc = close_curve(x, y)
    idx = np.lexsort((yc[:-1], xc[:-1]))[0]
    xr = np.r_[xc[idx:-1], xc[:idx]]
    yr = np.r_[yc[idx:-1], yc[:idx]]
    return xr, yr

def canonicalize_pair_open(x1, y1, x2, y2):
    """
    Usa el origen canónico de la PRIMERA curva y aplica la misma rotación a la SEGUNDA.
    Ambas curvas deben tener la misma longitud.
    """
    xc, yc = close_curve(x1, y1)
    idx = np.lexsort((yc[:-1], xc[:-1]))[0]
    def rot(x, y):
        x = np.r_[x, x[0]]; y = np.r_[y, y[0]]
        xr = np.r_[x[idx:-1], x[:idx]]
        yr = np.r_[y[idx:-1], y[:idx]]
        return xr[:-1], yr[:-1]
    x1o, y1o = rot(x1, y1)
    x2o, y2o = rot(x2, y2)
    return x1o, y1o, x2o, y2o

def efd_amplitude_spectrum(coeffs_full: np.ndarray) -> np.ndarray:
    """Amplitud por armónico: sqrt(A^2+B^2+C^2+D^2)."""
    return np.sqrt((coeffs_full**2).sum(axis=1))

def cumulative_fourier_power_like(coeffs_full: np.ndarray) -> np.ndarray:
    """Potencia acumulada fraccional ~ sum_{k<=n} k^2*(A^2+B^2+C^2+D^2) / total."""
    H = coeffs_full.shape[0]
    n = np.arange(1, H+1, dtype=float)
    p = (n**2) * (coeffs_full**2).sum(axis=1)
    c = np.cumsum(p)
    return c / c[-1]


# -----------------------------------------------------------------------------
# Contorno desde imagen binaria (rápido: sin Otsu ni morfología si assume_binary=True)
# -----------------------------------------------------------------------------
def contour_from_binary_image(path: str, assume_binary: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lee la imagen y extrae el contorno más largo con marching squares.
    Si `assume_binary=True`, considera foreground = pixeles > 0 (o > 0.5 si float).
    No se reduce resolución.
    Devuelve x,y *abiertas* y canonicalizadas.
    """
    img = io.imread(path)
    if img.ndim == 3:
        if _HAS_RGB2GRAY:
            img = rgb2gray(img)
        else:
            # Conversión manual simple a gris
            img = (0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]).astype(np.float32)
            vmin, vmax = float(img.min()), float(img.max())
            if vmax > vmin:
                img = (img - vmin) / (vmax - vmin)

    if assume_binary:
        # Booleana rápida: todo pixel >0 es foreground (soporta 0/255, 0/1, etc.)
        binm = img > 0
    else:
        # Fallback muy simple: normaliza y umbral 0.5 (evita Otsu para no recorrer histograma costoso)
        vmin, vmax = float(img.min()), float(img.max())
        img01 = (img - vmin) / (vmax - vmin + 1e-12)
        binm = img01 >= 0.5

    # Contornos al nivel 0.5
    contours = measure.find_contours(binm.astype(float), level=0.5)
    if not contours:
        raise RuntimeError("No se encontraron contornos en la imagen.")
    cnt = max(contours, key=lambda c: c.shape[0])  # contorno más largo
    y = cnt[:, 0]; x = cnt[:, 1]  # (fila, col) -> (y, x)
    x, y = canonicalize_open(x, y)
    return x, y


# -----------------------------------------------------------------------------
# Sintético "mancha" con lobos + concavidad (poco ruido fino)
# -----------------------------------------------------------------------------
def superellipse(theta: np.ndarray, a: float, b: float, m: float) -> np.ndarray:
    ct = np.cos(theta); st = np.sin(theta)
    return ((np.abs(ct)/a)**m + (np.abs(st)/b)**m) ** (-1.0/m)

def raised_cosine(x: np.ndarray, width: float) -> np.ndarray:
    z = np.zeros_like(x)
    mask = np.abs(x) <= (width/2)
    z[mask] = 0.5*(1 + np.cos(np.pi * x[mask] / (width/2)))
    return z

def blob_lobed_concave(M: int = 8192,
                       center: Tuple[float, float] = (320.0, 240.0),
                       R0: float = 160.0,
                       ecc: float = 0.18,
                       m_super: float = 3.0,
                       lobes: Iterable[Tuple[int, float]] = ((3, 0.12), (5, 0.08)),
                       bay_depth: float = 0.42,
                       bay_width_frac: float = 0.24,
                       seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera una mancha suave con lobos de baja frecuencia y una concavidad ancha.
    Sin jitter fino. Devuelve curva *abierta* y canonicalizada.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2*np.pi, M, endpoint=False)

    a = R0*(1+ecc); b = R0*(1-ecc)
    r = superellipse(theta, a=a, b=b, m=m_super)
    for k, amp in lobes:
        phi = rng.uniform(0, 2*np.pi)
        r *= (1.0 + amp*np.cos(k*theta + phi))

    theta0 = rng.uniform(0, 2*np.pi)
    width = bay_width_frac * 2*np.pi
    w = raised_cosine(np.angle(np.exp(1j*(theta - theta0))), width)
    r -= bay_depth * R0 * w

    r = np.maximum(r, 8.0)
    cx, cy = center
    x = cx + r*np.cos(theta); y = cy + r*np.sin(theta)

    x, y = canonicalize_open(x, y)
    # Asegurar orientación CCW (área positiva): si no, invertir
    xc, yc = close_curve(x, y)
    area = 0.5 * float(np.sum(xc[:-1]*yc[1:] - xc[1:]*yc[:-1]))
    if area < 0: x, y = x[::-1], y[::-1]
    return x, y


# -----------------------------------------------------------------------------
# EFD no normalizado: una sola vez hasta Nyquist; luego reconstrucciones por H
# -----------------------------------------------------------------------------
@dataclass
class EFDFull:
    coeffs_full: np.ndarray  # (H_max, 4)
    nyquist: int
    locus: Tuple[float, float]

def fit_efd_full(x: np.ndarray, y: np.ndarray) -> EFDFull:
    """Calcula coeficientes EFD hasta Nyquist y guarda el locus DC para reconstrucción en coords originales."""
    H_max = int(sefd.Nyquist(x))
    coeffs_full = sefd.CalculateEFD(x, y, H_max)
    locus = sefd.calculate_dc_coefficients(x, y)
    return EFDFull(coeffs_full=coeffs_full, nyquist=H_max, locus=(float(locus[0]), float(locus[1])))

def reconstruct_open(fit: EFDFull, H_use: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruye la curva con 'harmonic=H_use' en la MISMA malla paramétrica t y longitud M.
    Devuelve curva *abierta*.
    """
    H_use = int(max(1, min(H_use, fit.coeffs_full.shape[0])))
    xt, yt = sefd.inverse_transform(fit.coeffs_full, harmonic=H_use, locus=fit.locus, n_coords=M)
    return xt, yt


# -----------------------------------------------------------------------------
# Selección automática de H_curve y H_base
# -----------------------------------------------------------------------------
def choose_H_curve(coeffs_full: np.ndarray, p_curve: float = 0.999):
    """Elige H_curve como el primer H con potencia acumulada ≥ p_curve (o Nyquist si no llega)."""
    cumP = cumulative_fourier_power_like(coeffs_full)
    H_max = coeffs_full.shape[0]
    H_curve = int(np.argmax(cumP >= p_curve) + 1) if np.any(cumP >= p_curve) else H_max
    return H_curve, cumP

def choose_H_base(coeffs_full: np.ndarray,
                  fit: EFDFull,
                  H_curve: int,
                  M: int,
                  method: str = "cum_power",
                  p_base: float = 0.95,
                  lam: float = 0.05) -> int:
    """
    Elige H_base:
      - 'cum_power': primer H con potencia acumulada ≥ p_base (forzado < H_curve y ≥3).
      - 'tradeoff' : minimiza J(H)=RMS(h_H) + lam*(H/H_curve) usando h(t) con baseline H.
    """
    H_max = coeffs_full.shape[0]
    H_curve = int(max(2, min(H_curve, H_max)))

    if method == "cum_power":
        cumP = cumulative_fourier_power_like(coeffs_full)
        H_base = int(np.argmax(cumP >= p_base) + 1) if np.any(cumP >= p_base) else max(3, H_curve//4)
        H_base = max(3, min(H_base, H_curve-1))
        return H_base

    # Método tradeoff (más caro)
    x_curve, y_curve = reconstruct_open(fit, H_use=H_curve, M=M)
    Hb_candidates = np.arange(3, H_curve, dtype=int)
    best_H, best_J = Hb_candidates[0], np.inf
    for Hb in Hb_candidates:
        xb, yb = reconstruct_open(fit, H_use=Hb, M=M)
        # Alinear por origen canónico de la BASE y aplicar igual rotación a CURVE
        xb, yb, x_curve_al, y_curve_al = canonicalize_pair_open(xb, yb, x_curve, y_curve)

        # Normales de la BASE
        dx = np.gradient(xb); dy = np.gradient(yb)
        speed = np.hypot(dx, dy); speed = np.where(speed == 0, 1e-12, speed)
        Nx, Ny = -dy/speed, dx/speed

        h = (x_curve_al - xb)*Nx + (y_curve_al - yb)*Ny
        rms = float(np.sqrt(np.mean(h**2)))
        J = rms + lam*(Hb/float(H_curve))
        if J < best_J:
            best_J, best_H = J, int(Hb)
    return int(best_H)


# -----------------------------------------------------------------------------
# Unroll EFD-only: h(t) = (curve - base) · N_base(t)
# -----------------------------------------------------------------------------
def unroll_efd_only(fit: EFDFull, H_base: int, H_curve: int, M: int):
    xb, yb = reconstruct_open(fit, H_use=H_base, M=M)
    xc, yc = reconstruct_open(fit, H_use=H_curve, M=M)

    # Alinear ambas por el origen canónico de la BASE (misma rotación)
    xb, yb, xc, yc = canonicalize_pair_open(xb, yb, xc, yc)

    # Normales de la BASE
    dx = np.gradient(xb); dy = np.gradient(yb)
    speed = np.hypot(dx, dy); speed = np.where(speed == 0, 1e-12, speed)
    Nx, Ny = -dy/speed, dx/speed

    h = (xc - xb)*Nx + (yc - yb)*Ny
    t = np.linspace(0.0, 1.0, h.size, endpoint=False)
    return t, h, (xb, yb), (xc, yc)


# -----------------------------------------------------------------------------
# Errores vs H (EFD vs EFD). Para ser rápido, muestreamos a lo sumo 200 Hs.
# -----------------------------------------------------------------------------
def error_vs_H(fit: EFDFull, H_curve: int, M: int):
    H_curve = int(H_curve)
    H_list = np.unique(np.linspace(1, H_curve, min(H_curve, 200), dtype=int))
    x_ref, y_ref = reconstruct_open(fit, H_use=H_curve, M=M)
    cx, cy = np.mean(x_ref), np.mean(y_ref)
    r_ref = np.hypot(x_ref-cx, y_ref-cy)
    errs = []
    for H in H_list:
        xt, yt = reconstruct_open(fit, H_use=H, M=M)
        r = np.hypot(xt-cx, yt-cy)
        errs.append(float(np.sqrt(np.mean((r - r_ref)**2))))  # RMS radial
    return H_list, np.array(errs)


# -----------------------------------------------------------------------------
# DEMO principal
# -----------------------------------------------------------------------------
def demo_v5(mode: str = "synthetic",
            image_path: Optional[str] = None,
            assume_binary: bool = True,
            outdir: str = "efd_v5_out",
            M: int = 8192,
            p_curve: float = 0.999,
            baseline_method: str = "cum_power",  # o 'tradeoff'
            p_base: float = 0.95,
            lam_tradeoff: float = 0.05,
            seed: int = 7):

    os.makedirs(outdir, exist_ok=True)

    # 1) Obtener contorno de entrada (abierto)
    if mode == "synthetic":
        x_in, y_in = blob_lobed_concave(M=M, seed=seed)
    elif mode == "image":
        if image_path is None:
            raise ValueError("Debes pasar image_path para mode='image'.")
        x_in, y_in = contour_from_binary_image(image_path, assume_binary=assume_binary)
        # Para overlay limpio en plots, canonicalizamos también el contorno de entrada
        x_in, y_in = canonicalize_open(x_in, y_in)
    else:
        raise ValueError("mode debe ser 'synthetic' o 'image'.")

    # 2) EFD no normalizado hasta Nyquist (una sola vez)
    fit = fit_efd_full(x_in, y_in)

    # 3) H óptimos
    H_curve_opt, cumP = choose_H_curve(fit.coeffs_full, p_curve=p_curve)
    H_base_opt = choose_H_base(fit.coeffs_full, fit, H_curve_opt, M,
                               method=baseline_method, p_base=p_base, lam=lam_tradeoff)

    # 4) Reconstrucciones para overlay (mismo t)
    xb, yb = reconstruct_open(fit, H_use=H_base_opt, M=M)
    xc, yc = reconstruct_open(fit, H_use=H_curve_opt, M=M)
    # Alinear baseline/curve y trazar contorno de entrada cerrado para comparación
    xb_al, yb_al, xc_al, yc_al = canonicalize_pair_open(xb, yb, xc, yc)
    x_in_c, y_in_c = close_curve(x_in, y_in)
    xb_c, yb_c = close_curve(xb_al, yb_al)
    xc_c, yc_c = close_curve(xc_al, yc_al)

    fig, ax = plt.subplots(figsize=(6.9, 6.6))
    ax.plot(x_in_c, y_in_c, lw=1.0, alpha=0.6, label="Contorno entrada")
    ax.plot(xb_c, yb_c, lw=1.6, label=f"Baseline EFD H={H_base_opt}")
    ax.plot(xc_c, yc_c, lw=1.6, label=f"Curva EFD H={H_curve_opt}")
    ax.set_aspect('equal'); ax.set_title(f"Overlay EFD-only — Nyquist={fit.nyquist}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "01_overlay.png"), dpi=170)

    # 5) Error vs H (muestreo hasta 200 puntos) + líneas en H_base/H_curve
    H_list, errs = error_vs_H(fit, H_curve=H_curve_opt, M=M)
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.2))
    ax2.plot(H_list, errs, lw=1.6)
    ymax = np.max(errs) if errs.size else 1.0
    ax2.axvline(H_base_opt, ls="--"); ax2.text(H_base_opt, ymax*0.92, "H_base", rotation=90, va="top")
    ax2.axvline(H_curve_opt, ls="--"); ax2.text(H_curve_opt, ymax*0.92, "H_curve", rotation=90, va="top")
    ax2.set_xlabel("Nº de armónicos H"); ax2.set_ylabel("Error RMS radial [u.c.]")
    ax2.set_title("Error de reconstrucción vs H (EFD-only)")
    fig2.tight_layout(); fig2.savefig(os.path.join(outdir, "02_error_vs_H.png"), dpi=170)

    # 6) Espectro EFD + potencia acumulada (con líneas)
    amps = efd_amplitude_spectrum(fit.coeffs_full)
    n_idx = np.arange(1, fit.coeffs_full.shape[0]+1)
    fig3, ax3 = plt.subplots(figsize=(6.8, 4.2))
    ax3.plot(n_idx, amps, lw=1.4); ax3.axvline(H_base_opt, ls="--"); ax3.axvline(H_curve_opt, ls="--")
    ax3.set_xlabel("Armónico n"); ax3.set_ylabel("Amplitud EFD ~ √(A²+B²+C²+D²)")
    ax3.set_title("Espectro de amplitud EFD")
    fig3.tight_layout(); fig3.savefig(os.path.join(outdir, "03_efd_spectrum.png"), dpi=170)

    fig4, ax4 = plt.subplots(figsize=(6.8, 4.2))
    ax4.plot(n_idx, cumP, lw=1.6); ax4.axvline(H_base_opt, ls="--"); ax4.axvline(H_curve_opt, ls="--")
    ax4.set_xlabel("Armónico n"); ax4.set_ylabel("Potencia acumulada (fracción)")
    ax4.set_ylim(0, 1.02); ax4.set_title("Potencia acumulada tipo Fourier")
    fig4.tight_layout(); fig4.savefig(os.path.join(outdir, "04_cumulative_power.png"), dpi=170)

    # 7) Unroll: h(t) con baseline H_base_opt
    t, h, (xb_al, yb_al), (xc_al, yc_al) = unroll_efd_only(fit, H_base=H_base_opt, H_curve=H_curve_opt, M=M)
    fig5, ax5 = plt.subplots(figsize=(7.6, 4.2))
    ax5.plot(t, h, lw=1.2)
    ax5.set_xlabel("Parámetro t ∈ [0,1)"); ax5.set_ylabel("h(t): desplazamiento normal [u.c.]")
    ax5.set_title(f"Unroll EFD-only — H_base={H_base_opt}, RMS={np.sqrt(np.mean(h**2)):.3f}, P2P={np.max(h)-np.min(h):.3f}")
    fig5.tight_layout(); fig5.savefig(os.path.join(outdir, "05_unroll_h.png"), dpi=170)

    # 8) Espectro de rugosidad de h(t)
    Hfft = np.fft.rfft(h - np.mean(h))
    k = np.fft.rfftfreq(h.size, d=1.0/h.size)
    fig6, ax6 = plt.subplots(figsize=(6.8, 4.2))
    ax6.plot(k, np.abs(Hfft), lw=1.2)
    ax6.set_xlim(0, np.max(k)); ax6.set_xlabel("Frecuencia k [ciclos por perímetro]"); ax6.set_ylabel("|FFT(h)|")
    ax6.set_title("Espectro de rugosidad de h(t)")
    fig6.tight_layout(); fig6.savefig(os.path.join(outdir, "06_roughness_fft.png"), dpi=170)

    # 9) Guardado de arrays
    np.savez(os.path.join(outdir, "results_v5_fast.npz"),
             x_in=x_in, y_in=y_in, coeffs=fit.coeffs_full, nyquist=fit.nyquist, locus=np.array(fit.locus),
             H_base_opt=np.array([H_base_opt]), H_curve_opt=np.array([H_curve_opt]),
             t=t, h=h, amps=amps, cumP=cumP, H_list=H_list, errs=errs)

    print(f"[OK] Nyquist={fit.nyquist} | H_curve={H_curve_opt} | H_base={H_base_opt} | M={M} | outdir={outdir}")


# Ejecutable sencillo
if __name__ == "__main__":
    # Ejemplo sintético
    #demo_v5(mode="synthetic", outdir="efd_v5_out", M=8192, p_curve=0.999, p_base=0.95)
    # Ejemplo imagen (descomentar y ajustar la ruta):
    demo_v5(
    mode="image",
    image_path=r"C:\Users\Marina\Downloads\Bin-P8139-190Oe-30ms-5Tw-99.tif",
    assume_binary=True,
    baseline_method="tradeoff",   # << usa el selector más equilibrado
    lam_tradeoff=0.05,            # podés ajustar este número
    outdir="efd_v5_out"
)
