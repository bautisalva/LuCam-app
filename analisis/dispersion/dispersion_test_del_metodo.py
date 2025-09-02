# -*- coding: utf-8 -*-
"""
Medición de desplazamientos radiales por fajas (rings) entre dos dominios binarios C1→C2.

Este módulo implementa:
  1) Generación de dominios de prueba (círculos y contornos rugosos correlacionados/no correlacionados).
  2) Construcción de fajas de expansión/retracción alrededor de C1 usando EDT (distance transform).
  3) Cómputo por faja de A_k (píxeles que cambiaron) y P_k (píxeles totales en la faja) y métricas:
       ū = Σ (A_k^+/P_k^+) − Σ (A_k^−/P_k^−)
       ⟨u²⟩ (serie con coeficientes explícitos; por defecto orden=1 → 2·Σ r_k(A_k/P_k))
       Var = ⟨u²⟩ − ū²
     Incluye también una métrica con curvatura media mediante P_ref = (P1+P2)/2.
  4) Una referencia “por radio efectivo” en el caso circular.
  5) Funciones de plot.
  6) Dos “experimentos” reproducibles:
       - main(): tres casos (A círculo, B/C rugosos) con tablas y plots de fajas.
       - sweep_plot_variance(): barrido Δr para expansión circular perfecta y comparación de varianzas.

IMPORTANTE:
- No se modificó el algoritmo ni el estilo de los gráficos respecto de tu código original.
- Se unificaron utilidades y se agregó documentación extensa para facilitar lectura y mantenimiento.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.draw import polygon
from scipy.ndimage import distance_transform_edt

# ============================================================
# (0) Utilidades generales
# ============================================================

def to_bool(img: np.ndarray, thr: float = 0) -> np.ndarray:
    """Convierte una imagen (numérica) a binaria aplicando umbral estricto (>thr)."""
    return (img.astype(float) > thr)

def longest_contour(img: np.ndarray):
    """Devuelve el contorno más largo (array Nx2 con [y,x]) o None si no hay contornos."""
    conts = find_contours(img.astype(float), level=0.5)
    return max(conts, key=lambda c: c.shape[0]) if conts else None

def contour_length(coords) -> float:
    """Longitud (perímetro) de un contorno cerrado dado por coords[:,(y,x)]."""
    if coords is None:
        return 0.0
    y, x = coords[:, 0], coords[:, 1]
    dy = np.diff(y, append=y[0]); dx = np.diff(x, append=x[0])
    return float(np.sum(np.hypot(dx, dy)))

# ============================================================
# (1) Dominios artificiales: círculos y rugosos correlacionados/no
# ============================================================

def make_circle(shape: tuple[int, int], center: tuple[int, int], r: int, value: int = 1) -> np.ndarray:
    """Imagen binaria con un disco lleno de radio r en 'center'. Devuelve uint8 {0, value}."""
    H, W = shape; cy, cx = center
    y = np.arange(H)[:, None]; x = np.arange(W)[None, :]
    return (((y - cy)**2 + (x - cx)**2) <= r**2).astype(np.uint8) * value

def _smooth_noise_on_circle(n_theta: int = 2048, amp: float = 2.0, smooth_frac: float = 0.04, seed=None) -> np.ndarray:
    """
    Ruido 1D suavizado por truncado espectral (FFT) para perturbar radios en θ.
    Se normaliza a desvío std=amp (si std>0).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n_theta)
    F = np.fft.rfft(x)
    kmax = int(len(F) * smooth_frac)
    F[kmax:] = 0
    y = np.fft.irfft(F, n=n_theta)
    y = y/np.std(y)*amp if np.std(y) > 0 else y
    return y

def _rasterize_polar_shape(shape: tuple[int, int], center: tuple[int, int],
                           r_theta: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Rasteriza un contorno polar r(θ) a máscara binaria (rellena)."""
    cy, cx = center
    xs = cx + r_theta*np.cos(thetas)
    ys = cy + r_theta*np.sin(thetas)
    rr, cc = polygon(ys, xs, shape=shape)
    img = np.zeros(shape, dtype=np.uint8); img[rr, cc] = 1
    return img

def make_rough_pair(shape: tuple[int, int], center: tuple[int, int],
                    r0_1: float, r0_2: float, correlated: bool = True,
                    amp: float = 2.0, seed: int = 0, n_theta: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera dos máscaras rugosas C1,C2 con radios r(θ)=r0±ruido(θ).
    - correlated=True: ambos comparten la MISMA perturbación → contornos con rugosidad correlacionada.
    - correlated=False: perturbaciones independientes.
    """
    th = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    if correlated:
        eps = _smooth_noise_on_circle(n_theta, amp=amp, smooth_frac=0.04, seed=seed)
        r1, r2 = r0_1 + eps, r0_2 + eps
    else:
        eps1 = _smooth_noise_on_circle(n_theta, amp=amp, smooth_frac=0.04, seed=seed)
        eps2 = _smooth_noise_on_circle(n_theta, amp=amp, smooth_frac=0.04, seed=seed+1)
        r1, r2 = r0_1 + eps1, r0_2 + eps2
    C1 = _rasterize_polar_shape(shape, center, r1, th)
    C2 = _rasterize_polar_shape(shape, center, r2, th)
    return C1, C2

def build_three_cases(shape: tuple[int, int] = (512, 512), center: tuple[int, int] | None = None,
                      r1: int = 80, r2: int = 100):
    """
    Construye tres pares (C1,C2):
      A) Círculos perfectos (Δr=r2−r1)
      B) Rugosos correlacionados
      C) Rugosos NO correlacionados
    """
    if center is None:
        center = (shape[0]//2, shape[1]//2)
    C1A = make_circle(shape, center, r1)
    C2A = make_circle(shape, center, r2)
    C1B, C2B = make_rough_pair(shape, center, r1, r2, correlated=True,  amp=2.0, seed=7)
    C1C, C2C = make_rough_pair(shape, center, r1, r2, correlated=False, amp=2.0, seed=11)
    return (("A) Círculos", C1A, C2A),
            ("B) Rugosos correl.", C1B, C2B),
            ("C) Rugosos no corr.", C1C, C2C))

# ============================================================
# (2) Fajas de expansión y retracción C1 → C2
# ============================================================

def _quantize_ring_index(d: np.ndarray, convention: str = "touch_last", eps: float = 1e-6) -> np.ndarray:
    """
    Cuantiza distancias (enteros de faja) según convención:
      - 'left_closed' : [k, k+1)   → floor(d+eps)
      - 'touch_last'  : (k−1, k]   → ceil(d−eps) − 1  (garantiza que la última faja “toque” C2)
    Retorna índices k≥0 (se recortan negativos a 0 por estabilidad numérica cerca de d≈0).
    """
    if convention == "left_closed":
        k = np.floor(d + eps).astype(int)
    elif convention == "touch_last":
        k = np.ceil(d - eps).astype(int) - 1
    else:
        raise ValueError("convention debe ser 'left_closed' o 'touch_last'")
    return np.maximum(k, 0)

def bands_from_C1_to_C2(C1: np.ndarray, C2: np.ndarray, thr: float = 0, convention: str = "touch_last") -> dict:
    """
    Construye etiquetas de faja por distancia desde C1:
      growth = C2 & ~C1  (expansión)
      shrink = C1 & ~C2  (retracción)
      k_out  = cuantización de EDT(~C1) fuera de C1
      k_in   = cuantización de EDT( C1)  dentro de C1
    Devuelve diccionario con máscaras y índices de faja.
    """
    C1b = (C1.astype(float) > thr); C2b = (C2.astype(float) > thr)
    growth = C2b & (~C1b)
    shrink = C1b & (~C2b)

    d_in  = distance_transform_edt(C1b)    # hacia adentro
    d_out = distance_transform_edt(~C1b)   # hacia afuera

    k_out = np.full(C1b.shape, -1, dtype=int)
    k_in  = np.full(C1b.shape, -1, dtype=int)

    k_out_vals = _quantize_ring_index(d_out[~C1b], convention=convention)
    k_in_vals  = _quantize_ring_index(d_in[C1b],  convention=convention)

    k_out[~C1b] = k_out_vals
    k_in[C1b]   = k_in_vals

    return dict(C1b=C1b, C2b=C2b, growth=growth, shrink=shrink, k_in=k_in, k_out=k_out)

# ============================================================
# (3) Ak y Pk por faja + series/métricas
# ============================================================

def ak_pk_by_ring_from_C1(bands: dict):
    """
    Para cada k:
      Exterior: Rk_out = {~C1 & k_out==k}   →  Pk_pos = |Rk_out|,   Ak_pos = |Rk_out ∩ growth|
      Interior: Rk_in  = { C1 & k_in ==k}   →  Pk_neg = |Rk_in |,    Ak_neg = |Rk_in  ∩ shrink|
    r_k = k + 1/2 (en ambos lados).
    """
    C1b, growth, shrink = bands["C1b"], bands["growth"], bands["shrink"]
    k_out, k_in = bands["k_out"], bands["k_in"]

    # Exterior
    max_ko = int(k_out[~C1b].max()) if (~C1b).any() else -1
    Lpos = max_ko + 1 if max_ko >= 0 else 0
    Ak_pos = np.zeros(Lpos, dtype=np.int64)
    Pk_pos = np.zeros(Lpos, dtype=np.int64)
    for k in range(Lpos):
        ring = (~C1b) & (k_out == k)          # faja de 1 px
        Pk_pos[k] = int(ring.sum())           # “todos los puntos”
        if Pk_pos[k] > 0:
            Ak_pos[k] = int((ring & growth).sum())  # “los que cambiaron”

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

def perimeters_P1_P2(C1: np.ndarray, C2: np.ndarray) -> tuple[float, float]:
    """Perímetros (longitudes de contorno más largo) de C1 y C2."""
    c1 = longest_contour(C1); c2 = longest_contour(C2)
    P1 = contour_length(c1);   P2 = contour_length(c2)
    return P1, P2

def u2_series_from_moments(rk_pos: np.ndarray, rk_neg: np.ndarray,
                           Ak_pos: np.ndarray, Ak_neg: np.ndarray,
                           Pk_pos: np.ndarray, Pk_neg: np.ndarray,
                           order: int = 10, mode: str = "explicit_pi_no_pref",
                           Pref: float | None = None, kappa_avg: float | None = None) -> dict:
    """
    Serie para ⟨u²⟩ basada en momentos M_m ≡ Σ r_k^m * (A_k / P_k^m) sumando lados + y −.
    ū = Σ(A_k^+/P_k^+) − Σ(A_k^−/P_k^−)

    mode:
      - "explicit_pi_no_pref": coeficiente m → 2 * (-2π)^(m-1)
      - "kappa_avg"          : coeficiente m → 2 * κ^(m-1), con κ=kappa_avg o κ=−2π/Pref
    """
    # ū con Pk adentro
    u_mean = float(
        np.sum(np.divide(Ak_pos, Pk_pos, where=(Pk_pos > 0), out=np.zeros_like(Ak_pos, dtype=float))) -
        np.sum(np.divide(Ak_neg, Pk_neg, where=(Pk_neg > 0), out=np.zeros_like(Ak_neg, dtype=float)))
    )

    # M_m de cada lado
    def _moment_side(rk, Ak, Pk, m):
        if rk.size == 0:
            return 0.0
        num = rk.astype(float) ** m
        den = Pk.astype(float) ** m
        frac = np.divide(Ak.astype(float), den, where=(den > 0), out=np.zeros_like(Ak, dtype=float))
        return float(np.sum(num * frac))

    Ms = []
    for m in range(1, order + 1):
        M_pos = _moment_side(rk_pos, Ak_pos, Pk_pos, m)
        M_neg = _moment_side(rk_neg, Ak_neg, Pk_neg, m)
        Ms.append(M_pos + M_neg)

    # coeficientes
    if mode == "explicit_pi_no_pref":
        coeffs = [2.0 * ((-2.0 * np.pi) ** (m - 1)) for m in range(1, order + 1)]
    elif mode == "kappa_avg":
        if kappa_avg is None:
            if Pref is None or not np.isfinite(Pref) or Pref <= 0:
                raise ValueError("Para mode='kappa_avg' pasá kappa_avg o Pref válido.")
            kappa = -2.0 * np.pi / float(Pref)
        else:
            kappa = float(kappa_avg)
        coeffs = [2.0 * (kappa ** (m - 1)) for m in range(1, order + 1)]
    else:
        raise ValueError("mode debe ser 'explicit_pi_no_pref' o 'kappa_avg'.")

    # <u^2> truncada
    u2_mean = float(sum(c * M for c, M in zip(coeffs, Ms)))
    var_u = u2_mean - u_mean ** 2  # no se recorta si fuese negativa
    return dict(u_mean=u_mean, u2_mean=u2_mean, var_u=var_u, Ms=Ms, coeffs=coeffs)

def dispersion_with_Pk_inside_and_curv(C1: np.ndarray, C2: np.ndarray, bands: dict) -> dict:
    """
    Métrica con P_k adentro y posibilidad de corrección de curvatura mediante P_ref=(P1+P2)/2.
    En esta implementación (idéntica a tu versión activa) se usa SOLO el término de orden 1:
       ⟨u²⟩ = 2 * Σ r_k (A_k/P_k)
    y se devuelven P1, P2, P_ref para referencia.
    """
    rk_pos, rk_neg, Ak_pos, Ak_neg, Pk_pos, Pk_neg = ak_pk_by_ring_from_C1(bands)

    # Perímetros y curvatura media con P_ref (se informa; no se usa en coeficientes por defecto)
    P1, P2 = perimeters_P1_P2(bands["C1b"], bands["C2b"])
    Pref = 0.5*(P1 + P2) if (P1+P2) > 0 else np.nan

    # Serie exactamente como en tu código (order=1, explicit_pi_no_pref)
    series = u2_series_from_moments(rk_pos, rk_neg, Ak_pos, Ak_neg, Pk_pos, Pk_neg,
                                    order=1, mode="explicit_pi_no_pref")

    return dict(u_mean=series["u_mean"], u2_mean=series["u2_mean"], var_u=series["var_u"],
                rk_pos=rk_pos, rk_neg=rk_neg,
                Ak_pos=Ak_pos, Ak_neg=Ak_neg, Pk_pos=Pk_pos, Pk_neg=Pk_neg,
                P1=P1, P2=P2, Pref=Pref)

# ============================================================
# (4) Método de radio efectivo (referencia: círculos)
# ============================================================

def radial_effective_dispersion_for_circles(C1: np.ndarray, C2: np.ndarray,
                                            center: tuple[int, int] | None = None,
                                            n_theta: int = 720, q: float = 0.90) -> dict:
    """
    Referencia para contornos circulares (o casi):
      - Obtiene radios por ángulo del contorno más largo de C1 y C2.
      - Re-muestrea en una malla uniforme de θ y usa el cuantil q por bin.
      - Devuelve ū, ⟨u²⟩ y Var de u(θ)=r2(θ)−r1(θ).

    Para círculos perfectos la media o el cuantil coinciden, así que no afecta resultados.
    """
    if center is None:
        center = (C1.shape[0]//2, C1.shape[1]//2)

    def contour_rt(img):
        cont = longest_contour(img)
        cy, cx = center
        y, x = cont[:, 0].astype(float), cont[:, 1].astype(float)
        dy, dx = y - cy, x - cx
        r = np.hypot(dx, dy)
        th = np.arctan2(dy, dx) % (2*np.pi)
        return th, r

    th1, r1 = contour_rt(C1)
    th2, r2 = contour_rt(C2)

    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    edges  = np.linspace(0, 2*np.pi, n_theta+1)

    def resample(th, r):
        bins = np.digitize(th, edges) - 1
        rs = np.full(n_theta, np.nan)
        for k in range(n_theta):
            vals = r[bins == k]
            if vals.size > 0:
                rs[k] = np.quantile(vals, q)
        valid = ~np.isnan(rs)
        if valid.any():
            t_ext = np.r_[thetas-2*np.pi, thetas, thetas+2*np.pi]
            r_ext = np.r_[rs, rs, rs]
            rs = np.where(valid, rs, np.interp(thetas, t_ext[~np.isnan(r_ext)], r_ext[~np.isnan(r_ext)]))
        return rs

    r1s = resample(th1, r1); r2s = resample(th2, r2)
    u = r2s - r1s
    return dict(u_mean=float(np.nanmean(u)),
                u2_mean=float(np.nanmean(u**2)),
                var_u=float(np.nanvar(u, ddof=0)))

# ============================================================
# (5) Plots
# ============================================================

def plot_bands(C1: np.ndarray, C2: np.ndarray, bands: dict, stats: dict, title: str = "") -> None:
    """
    Visualiza:
      (i) mapas de r_k en crecimiento/retracción sobre C1/C2,
      (ii) barras de A_k, P_k y A_k/P_k versus ±r_k.
    """
    C1b, C2b = bands["C1b"], bands["C2b"]
    k_out, k_in = bands["k_out"], bands["k_in"]
    growth, shrink = bands["growth"], bands["shrink"]

    # Mapas de r_k (solo donde hay cambio)
    map_out = np.full(C1b.shape, np.nan, dtype=float)
    map_in  = np.full(C1b.shape, np.nan, dtype=float)
    map_out[growth] = k_out[growth] + 0.5
    map_in[shrink]  = k_in[shrink] + 0.5

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axs[0].imshow(map_out, cmap="Greens", origin="upper")
    axs[0].contour(C1b, [0.5], colors="white", linewidths=1)
    axs[0].contour(C2b, [0.5], colors="black", linewidths=1)
    axs[0].set_title("Crecimiento: fajas y r_k"); axs[0].axis("off")
    c0 = plt.colorbar(im0, ax=axs[0]); c0.set_label("r_k")

    im1 = axs[1].imshow(map_in, cmap="Reds", origin="upper")
    axs[1].contour(C1b, [0.5], colors="white", linewidths=1)
    axs[1].contour(C2b, [0.5], colors="black", linewidths=1)
    axs[1].set_title("Retracción: fajas y r_k"); axs[1].axis("off")
    c1 = plt.colorbar(im1, ax=axs[1]); c1.set_label("r_k")
    plt.suptitle(title); plt.tight_layout(); plt.show()

    # Barras Ak, Pk y Ak/Pk en el MISMO eje de r_k
    rk_pos, rk_neg = stats["rk_pos"], stats["rk_neg"]
    Ak_pos, Ak_neg = stats["Ak_pos"], stats["Ak_neg"]
    Pk_pos, Pk_neg = stats["Pk_pos"], stats["Pk_neg"]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # A_k
    axs[0].bar(rk_pos, Ak_pos, width=0.9, color="green", alpha=0.7, label="Ak+")
    axs[0].bar(-rk_neg, Ak_neg, width=0.9, color="red",   alpha=0.7, label="Ak-")
    axs[0].set_title("Conteos por faja (A_k)"); axs[0].legend()
    axs[0].set_xlabel("± r_k"); axs[0].set_ylabel("# píxeles")

    # P_k
    axs[1].bar(rk_pos, Pk_pos, width=0.9, color="green", alpha=0.7, label="Pk+")
    axs[1].bar(-rk_neg, Pk_neg, width=0.9, color="red",   alpha=0.7, label="Pk-")
    axs[1].set_title("Tamaño de faja (P_k)"); axs[1].legend()
    axs[1].set_xlabel("± r_k"); axs[1].set_ylabel("# píxeles")
    # Eje x simétrico y limitado al rango observado
    xmax = max(rk_pos.max() if rk_pos.size else 0, rk_neg.max() if rk_neg.size else 0)
    axs[0].set_xlim(-xmax-1, xmax+1); axs[1].set_xlim(-xmax-1, xmax+1)

    # Ak/Pk
    ratio_pos = Ak_pos / np.maximum(Pk_pos, 1)
    ratio_neg = Ak_neg / np.maximum(Pk_neg, 1)
    axs[2].bar(rk_pos, ratio_pos, width=0.9, color="green", alpha=0.7, label="Ak+/Pk+")
    axs[2].bar(-rk_neg, ratio_neg, width=0.9, color="red",   alpha=0.7, label="Ak-/Pk-")
    axs[2].set_ylim(0, 1.05*max(1e-9, np.nanmax([ratio_pos.max() if ratio_pos.size else 0,
                                                 ratio_neg.max() if ratio_neg.size else 0])))
    axs[2].set_xlim(-xmax-1, xmax+1)
    axs[2].set_title("Fracción desplazada por faja (Ak/Pk)"); axs[2].legend()
    axs[2].set_xlabel("± r_k"); axs[2].set_ylabel("fracción")

    plt.tight_layout(); plt.show()

# ============================================================
# (6) Segundo set de utilidades para “experimento Δr”
#     (idéntico en comportamiento a tu bloque final)
# ============================================================

def make_circle_basic(shape: tuple[int, int], center: tuple[int, int], r: int) -> np.ndarray:
    """Versión mínima (para barrido Δr); equivalente a make_circle(shape,center,r, value=1)."""
    H, W = shape; cy, cx = center
    y = np.arange(H)[:, None]; x = np.arange(W)[None, :]
    return (((y - cy)**2 + (x - cx)**2) <= r**2).astype(np.uint8)

def bands_from_C1_to_C2_basic(C1: np.ndarray, C2: np.ndarray) -> dict:
    """
    Variante mínima del etiquetado de fajas usada en el experimento de barrido.
    Aplica la convención “touch_last”.
    """
    C1b = C1.astype(bool); C2b = C2.astype(bool)
    growth = C2b & (~C1b)   # expansión
    shrink = C1b & (~C2b)   # retracción

    d_in  = distance_transform_edt(C1b)    # interior
    d_out = distance_transform_edt(~C1b)   # exterior

    k_out = np.full(C1b.shape, -1, dtype=int)
    k_in  = np.full(C1b.shape, -1, dtype=int)
    k_out[~C1b] = _quantize_ring_index(d_out[~C1b], convention="touch_last")
    k_in[C1b]   = _quantize_ring_index(d_in[C1b],  convention="touch_last")

    return dict(C1b=C1b, C2b=C2b, growth=growth, shrink=shrink, k_out=k_out, k_in=k_in)

def ak_pk_by_ring_from_C1_basic(bands: dict):
    """Mismo comportamiento que ak_pk_by_ring_from_C1; se mantiene por claridad del experimento."""
    return ak_pk_by_ring_from_C1(bands)

def perimeters_P1_P2_basic(C1: np.ndarray, C2: np.ndarray) -> tuple[float, float]:
    """Igual que perimeters_P1_P2 (compat para el experimento)."""
    return perimeters_P1_P2(C1, C2)

def dispersion_pk(C1: np.ndarray, C2: np.ndarray, use_curvature: bool = False) -> tuple[float, float, float]:
    """
    Devuelve u_mean, u2_mean, Var usando P_k dentro de la fracción (Ak/Pk).
    Si use_curvature=True, añade término:  −(4π/Pref) * Σ r_k² (Ak/Pk),
    con Pref = (P1+P2)/2.
    """
    bands = bands_from_C1_to_C2_basic(C1, C2)
    rk_pos, rk_neg, Ak_pos, Ak_neg, Pk_pos, Pk_neg = ak_pk_by_ring_from_C1_basic(bands)

    mpos = Pk_pos > 0
    mneg = Pk_neg > 0

    # u medio
    u_mean = float((Ak_pos[mpos]/Pk_pos[mpos]).sum() - (Ak_neg[mneg]/Pk_neg[mneg]).sum())

    # 2 * Σ r_k (Ak/Pk)
    M1 = float((rk_pos[mpos] * (Ak_pos[mpos]/Pk_pos[mpos])).sum()
             + (rk_neg[mneg] * (Ak_neg[mneg]/Pk_neg[mneg])).sum())
    u2_mean = 2.0 * M1

    if use_curvature:
        # - (4π/Pref) * Σ r_k^2 (Ak/Pk)
        P1, P2 = perimeters_P1_P2_basic(C1, C2)
        Pref = 0.5*(P1 + P2) if (P1 + P2) > 0 else np.nan
        M2 = float(((rk_pos[mpos]**2) * (Ak_pos[mpos]/Pk_pos[mpos])).sum()
                 + ((rk_neg[mneg]**2) * (Ak_neg[mneg]/Pk_neg[mneg])).sum())
        if np.isfinite(Pref) and Pref > 0:
            u2_mean += (-4.0*np.pi/Pref) * M2

    var_u = u2_mean - u_mean**2
    return u_mean, u2_mean, var_u

def radial_dispersion_reference(C1: np.ndarray, C2: np.ndarray,
                                center: tuple[int, int] | None = None,
                                n_theta: int = 720) -> tuple[float, float, float]:
    """
    Referencia radial “media por bin de θ”.
    Para círculos perfectos es exacto (coincide con el método de cuantiles).
    """
    if center is None:
        center = (C1.shape[0]//2, C1.shape[1]//2)

    def contour_rt(img):
        cont = longest_contour(img)
        y, x = cont[:, 0].astype(float), cont[:, 1].astype(float)
        dy, dx = y - center[0], x - center[1]
        r = np.hypot(dx, dy); th = np.arctan2(dy, dx) % (2*np.pi)
        return th, r

    th1, r1 = contour_rt(C1); th2, r2 = contour_rt(C2)
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    edges  = np.linspace(0, 2*np.pi, n_theta+1)

    def resample(th, r):
        bins = np.digitize(th, edges) - 1
        rs = np.full(n_theta, np.nan)
        for k in range(n_theta):
            vals = r[bins == k]
            if vals.size > 0:
                rs[k] = np.mean(vals)  # media (círculos perfectos → exacto)
        valid = ~np.isnan(rs)
        if valid.any():
            t_ext = np.r_[thetas-2*np.pi, thetas, thetas+2*np.pi]
            r_ext = np.r_[rs, rs, rs]
            rs = np.where(valid, rs, np.interp(thetas, t_ext[~np.isnan(r_ext)], r_ext[~np.isnan(r_ext)]))
        return rs

    r1s = resample(th1, r1); r2s = resample(th2, r2)
    u = r2s - r1s
    u_mean = float(np.nanmean(u))
    u2_mean = float(np.nanmean(u**2))
    var_u = float(np.nanvar(u, ddof=0))
    return u_mean, u2_mean, var_u

# ============================================================
# (7) Experimentos y visualizaciones de referencia
# ============================================================

def main() -> None:
    """
    Caso de estudio con tres pares (A, B, C). Imprime tabla y muestra:
      - mapas de fajas r_k en crecimiento/retracción
      - barras de Ak, Pk y Ak/Pk
    También compara con la referencia de radio efectivo SOLO para el caso circular.
    """
    shape=(512,512); center=(256,256)
    cases = build_three_cases(shape, center, r1=80, r2=100)  # Δr = 20

    print("\n=== Dispersión con P_k adentro + curvatura (P_ref=(P1+P2)/2) ===")
    print(f"{'Caso':22s} {'ū(Pk)':>9s} {'⟨u²⟩(Pk)':>12s} {'Var(Pk)':>12s} {'P1':>8s} {'P2':>8s} {'Pref':>8s}")

    for label, C1, C2 in cases:
        bands = bands_from_C1_to_C2(C1, C2)
        stats = dispersion_with_Pk_inside_and_curv(C1, C2, bands)
        plot_bands(C1, C2, bands, stats, title=label)

        print(f"{label:22s} {stats['u_mean']:9.3f} {stats['u2_mean']:12.3f} {stats['var_u']:12.3f} "
              f"{stats['P1']:8.1f} {stats['P2']:8.1f} {stats['Pref']:8.1f}")

    # Comparación con radio efectivo SOLO en el caso circular:
    label, C1, C2 = cases[0]
    ref = radial_effective_dispersion_for_circles(C1, C2, center=center)
    print("\n=== Referencia (círculos, radio efectivo) ===")
    print(f"ū(rad) = {ref['u_mean']:.3f}   ⟨u²⟩(rad) = {ref['u2_mean']:.3f}   Var(rad) = {ref['var_u']:.3f}")

def sweep_plot_variance(shape: tuple[int, int] = (512,512), r1: int = 80,
                        dr_list: np.ndarray | None = None, use_curvature: bool = False) -> None:
    """
    Barrido Δr para expansión radial perfecta. Compara:
      - Var (fajas, Pk adentro)
      - Var (fajas + curvatura) [opcional]
      - Var (radio efectivo)
    """
    if dr_list is None:
        dr_list = np.arange(1, 61)  # Δr de 1 a 60

    center = (shape[0]//2, shape[1]//2)

    var_pk_list = []
    var_pk_curv_list = []
    var_ref_list = []

    for dr in dr_list:
        C1 = make_circle_basic(shape, center, r1)
        C2 = make_circle_basic(shape, center, r1 + int(dr))

        # Pk “adentro” (sin curvatura)
        _, _, var_pk = dispersion_pk(C1, C2, use_curvature=False)
        var_pk_list.append(var_pk)

        # Pk “adentro” (con curvatura, si se pide)
        if use_curvature:
            _, _, var_pk_curv = dispersion_pk(C1, C2, use_curvature=True)
            var_pk_curv_list.append(var_pk_curv)

        # Referencia radial
        _, _, var_ref = radial_dispersion_reference(C1, C2, center=center)
        var_ref_list.append(var_ref)

    # ---- Plot ----
    plt.figure(figsize=(8,5))
    plt.plot(dr_list, var_pk_list, marker='o', linewidth=1.2, label='Var (fajas, Pk adentro)')
    if use_curvature:
        plt.plot(dr_list, var_pk_curv_list, marker='s', linewidth=1.2, label='Var (fajas + curvatura)')
    plt.plot(dr_list, var_ref_list, marker='^', linewidth=1.2, label='Var (radio efectivo)')
    plt.xlabel('Δr (pix)')
    plt.ylabel('Varianza')
    plt.title('Expansión radial perfecta: Varianza vs Δr')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# (8) Entradas “script”: se mantiene el orden de ejecución original
# ============================================================

if __name__ == "__main__":
    # Ejecuta el estudio de tres casos y luego el barrido Δr, tal como en tu script original.
    main()
    # Cambiá use_curvature=True si querés incluir el término −4π/Pref * Σ r_k^2 (A_k/P_k).
    sweep_plot_variance(shape=(512,512), r1=80, dr_list=np.arange(1, 61), use_curvature=False)

# ============================================================
# (9) Lista de PARÁMETROS IMPORTANTES (para edición rápida)
#    (No usados como “config” automática: solo guía de edición)
# ============================================================
"""
Generación de datos:
- shape:           (512, 512)     # tamaño de la imagen
- center:          (256, 256)     # centro geométrico por defecto
- r1, r2:          80, 100        # radios base para casos A/B/C (Δr = r2 − r1)
- n_theta:         2048 (rugosos) / 720 (referencias)  # resolución angular
- amp:             2.0            # amplitud de rugosidad
- smooth_frac:     0.04           # fracción de espectro retenida en suavizado
- seed:            7 / 11 / ...   # semillas para reproducibilidad en rugosos
- value:           1              # valor de “píxel ON” en máscaras binarias

Fajas y métricas:
- convention:      "touch_last"   # cuantización por defecto (ver _quantize_ring_index)
- thr:             0              # umbral para convertir a binario (to_bool / bands)
- order:           1              # orden de la serie en u2_series_from_moments (por defecto 1)
- mode:            "explicit_pi_no_pref"  # coeficientes (alternativa: "kappa_avg")
- Pref:            (P1+P2)/2      # perímetro de referencia (si se usa mode="kappa_avg")
- kappa_avg:       None           # curvatura media directa (si preferís fijarla)

Referencia radial:
- q:               0.90           # cuantil por bin de θ en el método “circles” (no afecta círculos perfectos)
- n_theta (ref):   720            # cantidad de bins angulares en la referencia

Experimentos:
- dr_list:         np.arange(1, 61)  # rango de Δr en el barrido
- use_curvature:   False              # incluir (o no) el término de curvatura en dispersion_pk
"""
