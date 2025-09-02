
"""
Dominios sintéticos tipo "manchas" + parametrización por Fourier del desplazamiento normal
=========================================================================================

Este script hace dos cosas principales:
1) Genera DOS dominios sintéticos tipo "mancha" (máscara binaria en píxeles) usando un
   radio con ruido angular (mismo estilo que el del código de referencia).
2) A partir del borde del dominio INICIAL, parametriza por longitud de arco s∈[0,1),
   calcula las normales "hacia afuera" y estima el DESPLAZAMIENTO normal u(s) que lleva
   el borde inicial al borde final. Luego analiza u(s) en Fourier y reconstruye.

Además:
- Ilustra las "fajas" (capas) de crecimiento/retracción conectadas al borde inicial.
- Grafica chequeos para validar la parametrización y la proyección normal.
- Guarda imágenes y (opcionalmente) un GIF de fajas si `matplotlib` tiene ffmpeg.

Requisitos: numpy, matplotlib, scipy, scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional

from scipy import ndimage as ndi
from skimage.morphology import binary_erosion, binary_dilation, square
from skimage.measure import find_contours

# -----------------------------------------------------------------------------
# 1) MANCHAS: generación y fajas (basado en tu estilo de "círculos ruidosos")
# -----------------------------------------------------------------------------

def noisy_circle_mask(size: int, r_base: float, noise_amp: float = 5.0,
                      n_points: int = 1024, seed: Optional[int] = None):
    """
    Genera una máscara binaria centro-radial con radio angular r(θ) = r_base + ruido(θ).

    Parámetros
    ----------
    size : lado de la imagen cuadrada (px)
    r_base : radio medio (px)
    noise_amp : amplitud del ruido angular (px)
    n_points : resolución angular para el perfil r(θ)
    seed : semilla RNG

    Devuelve
    --------
    mask : (H,W) uint8
    r_interp : (H,W) float, radio objetivo interpolado por ángulo
    theta_grid : (H,W) float, ángulo de cada píxel en [0,2π)
    """
    if seed is not None:
        np.random.seed(seed)

    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2
    dx, dy = x - cx, y - cy
    r_grid = np.hypot(dx, dy)

    # Perfil angular r(θ) suave, combinación de senos con fases aleatorias
    theta_1d = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    f = np.sin(3*theta_1d + np.random.uniform(0, 2*np.pi))
    f += 0.5*np.sin(7*theta_1d + np.random.uniform(0, 2*np.pi))
    f += 0.3*np.sin(13*theta_1d + np.random.uniform(0, 2*np.pi))
    f /= np.max(np.abs(f))

    r_profile = r_base + noise_amp * f
    # hacemos circular la interpolación en el extremo
    r_profile_ext = np.concatenate([r_profile, [r_profile[0]]])
    theta_ext = np.concatenate([theta_1d, [2*np.pi]])

    theta_grid = np.arctan2(dy, dx) % (2*np.pi)
    r_interp = np.interp(theta_grid.ravel(), theta_ext, r_profile_ext).reshape(theta_grid.shape)

    mask = (r_grid <= r_interp).astype(np.uint8)
    return mask, r_interp, theta_grid


def decompose_bands_from_edge(mask_initial: np.ndarray,
                              mask_final: np.ndarray,
                              selem=None):
    """
    Devuelve las capas (fajas) conectadas al borde inicial que suman crecimiento
    (y retracción) hasta llegar al dominio final.

    Retorna:
      thickness: mapa de "espesor" por capas (incluye el borde inicial como capa central)
      layers_for_anim: lista de capas booleanas SIN incluir el borde (óptimo para animación)
      edge_initial: borde inicial booleano
    """
    if selem is None:
        selem = square(3)

    growth = (mask_final == 1) & (mask_initial == 0)
    shrink = (mask_initial == 1) & (mask_final == 0)

    edge_initial = mask_initial.astype(bool) & (~binary_erosion(mask_initial, selem))

    # crecer desde el borde
    layers_growth: List[np.ndarray] = []
    canvas = edge_initial.copy()
    remaining = growth.copy()
    while remaining.any():
        new_layer = remaining & binary_dilation(canvas, selem)
        if not new_layer.any():
            break
        layers_growth.append(new_layer.copy())
        canvas |= new_layer
        remaining &= ~new_layer

    # retraer hacia el borde
    layers_shrink: List[np.ndarray] = []
    canvas_shrink = edge_initial.copy()
    remaining_shrink = shrink.copy()
    while remaining_shrink.any():
        new_layer = remaining_shrink & binary_dilation(canvas_shrink, selem)
        if not new_layer.any():
            break
        layers_shrink.append(new_layer.copy())
        canvas_shrink |= new_layer
        remaining_shrink &= ~new_layer

    # para animación excluimos el borde
    layers_for_anim = layers_growth + layers_shrink[::-1]

    # Para mapa thickness sí incluimos el borde
    layers_all_for_thickness = layers_growth + [edge_initial] + layers_shrink[::-1]
    thickness = np.zeros_like(mask_initial, dtype=int)
    for i, layer in enumerate(layers_all_for_thickness):
        thickness[layer] = i + 1

    return thickness, layers_for_anim, edge_initial


# -----------------------------------------------------------------------------
# 2) BORDE + PARAMETRIZACIÓN POR LONGITUD DE ARCO
# -----------------------------------------------------------------------------

def extract_ordered_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Extrae un contorno cerrado como lista ORDENADA de puntos (x,y) flotantes.

    Usa skimage.measure.find_contours a nivel 0.5 y toma el contorno más largo.
    """
    contours = find_contours(mask.astype(float), 0.5)
    if len(contours) == 0:
        return np.zeros((0, 2), dtype=float)

    # Elegimos el contorno más largo
    cnt = max(contours, key=lambda c: c.shape[0])
    # find_contours devuelve en (fila, col) = (y, x)
    xy = np.stack([cnt[:, 1], cnt[:, 0]], axis=1)
    return xy


def resample_by_arclength(xy: np.ndarray, n: int) -> np.ndarray:
    """
    Reparametriza una curva cerrada (ordenada) por longitud de arco y remuestrea
    a n puntos uniformes en s∈[0,1).
    """
    if xy.shape[0] < 2:
        return xy
    pts = np.vstack([xy, xy[0]])
    seg = np.diff(pts, axis=0)
    ds = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0], np.cumsum(ds)])
    total = s[-1]
    if total == 0:
        return np.tile(xy[:1], (n, 1))
    s_u = np.linspace(0, total, n, endpoint=False)
    x = np.interp(s_u, s, pts[:, 0])
    y = np.interp(s_u, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def tangent_and_normal(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tangentes y normales unitarias de una curva cerrada discreta (derivada central).
    """
    d = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    t = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    nvec = np.stack([-t[:, 1], t[:, 0]], axis=1)  # rotación +90°
    return t, nvec


def orient_normals_outward(xy: np.ndarray, nvec: np.ndarray, mask: np.ndarray,
                           eps: float = 0.75) -> np.ndarray:
    """
    Orienta las normales para que apunten hacia "afuera" del dominio dado por `mask`.
    """
    H, W = mask.shape
    out = nvec.copy()
    for i, p in enumerate(xy):
        probe = p + eps * out[i]
        x, y = int(round(probe[0])), int(round(probe[1]))
        inside = False
        if 0 <= x < W and 0 <= y < H:
            inside = mask[y, x] > 0
        # si el paso va hacia adentro, invertimos
        if inside:
            out[i] = -out[i]
    return out


# -----------------------------------------------------------------------------
# 3) DESPLAZAMIENTO NORMAL u(s) ENTRE DOS DOMINIOS
# -----------------------------------------------------------------------------

def bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Interpolación bilineal en (x,y) flotantes. Fuera de imagen -> borde más cercano.
    """
    H, W = img.shape
    xs = np.clip(xs, 0, W - 1 - 1e-6)
    ys = np.clip(ys, 0, H - 1 - 1e-6)
    x0 = np.floor(xs).astype(int); x1 = x0 + 1
    y0 = np.floor(ys).astype(int); y1 = y0 + 1
    wx = xs - x0; wy = ys - y0
    Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
    return (Ia * (1 - wx) * (1 - wy) +
            Ib * wx * (1 - wy) +
            Ic * (1 - wx) * wy +
            Id * wx * wy)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """
    Distancia con signo: positiva adentro, negativa afuera.
    """
    d_out = ndi.distance_transform_edt(~mask.astype(bool))
    d_in = ndi.distance_transform_edt(mask.astype(bool))
    return d_out - d_in  # >0 inside, <0 outside


def normal_ray_intersection_u(xy0: np.ndarray, n_out: np.ndarray,
                              sdf_final: np.ndarray,
                              tmax: float = 60.0, step: float = 0.5) -> np.ndarray:
    """
    Para cada punto del borde inicial xy0 y su normal "hacia afuera" n_out,
    busca el cruce sdf_final=0 a lo largo del rayo p + t*n_out.

    Devuelve u(s) = t* con signo (px). Si no encuentra cruce, deja NaN.
    """
    H, W = sdf_final.shape
    us = np.full(xy0.shape[0], np.nan, dtype=float)
    ts = np.arange(0.0, tmax + step, step)  # solo hacia afuera
    for i, p in enumerate(xy0):
        xs = p[0] + ts * n_out[i, 0]
        ys = p[1] + ts * n_out[i, 1]
        vals = bilinear_sample(sdf_final, xs, ys)
        # buscamos el primer cambio de signo o el mínimo en valor absoluto
        sign = np.sign(vals)
        zero_idx = np.where(np.abs(vals) < 1e-2)[0]
        if zero_idx.size > 0:
            us[i] = ts[zero_idx[0]]
            continue
        # si no hay casi-cero, usamos primer lugar donde sign cambia de + a - o viceversa
        sgn_change = np.where(np.diff(sign) != 0)[0]
        if sgn_change.size > 0:
            j = sgn_change[0]
            # interpolación lineal en el segmento [j, j+1]
            t0, t1 = ts[j], ts[j+1]
            v0, v1 = vals[j], vals[j+1]
            if v1 != v0:
                us[i] = t0 - v0 * (t1 - t0) / (v1 - v0)
            else:
                us[i] = t0
        else:
            # fallback: tomamos el mínimo |sdf| como aproximación
            j = int(np.argmin(np.abs(vals)))
            if np.isfinite(vals[j]):
                us[i] = ts[j]
    return us


# -----------------------------------------------------------------------------
# 4) FOURIER sobre u(s) y reconstrucción
# -----------------------------------------------------------------------------

def fft_spectrum(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Espectro de potencia de u(s) usando FFT real.
    Retorna k (índices de modo) y |U_k|^2.
    """
    N = len(u)
    U = np.fft.rfft(u - np.nanmean(u))
    power = np.abs(U) ** 2
    k = np.arange(U.size)
    return k, power


def reconstruct_from_fft(u: np.ndarray, K: int) -> np.ndarray:
    """
    Reconstruye u(s) reteniendo solo K modos de alta energía (rfft bins 0..K).
    """
    u0 = u.copy()
    mean = np.nanmean(u0)
    u0 = np.nan_to_num(u0 - mean)
    U = np.fft.rfft(u0)
    U[K+1:] = 0.0
    u_rec = np.fft.irfft(U, n=len(u0)) + mean
    return u_rec


# -----------------------------------------------------------------------------
# 5) DEMO PRINCIPAL
# -----------------------------------------------------------------------------

@dataclass
class Params:
    SIZE: int = 512
    R1: float = 100.0
    R2: float = 140.0
    NOISE1: float = 12.0
    NOISE2: float = 15.0
    SELEM_SIZE: int = 3
    N_CURVE: int = 1500
    TMAX_RAY: float = 70.0
    STEP_RAY: float = 0.5
    K_RECON: int = 25


def main():
    P = Params()

    # --- Generar dominios "mancha" ---
    mask1, _, _ = noisy_circle_mask(P.SIZE, P.R1, P.NOISE1, seed=1)
    mask2, _, _ = noisy_circle_mask(P.SIZE, P.R2, P.NOISE2, seed=2)

    # --- Fajas desde el borde inicial ---
    thickness, layers, edge_initial = decompose_bands_from_edge(mask1, mask2, square(P.SELEM_SIZE))

    # --- Extraer borde ordenado y parametrizar por arco ---
    boundary0 = extract_ordered_boundary(mask1)
    boundary0_u = resample_by_arclength(boundary0, P.N_CURVE)
    s = np.linspace(0, 1, P.N_CURVE, endpoint=False)

    # --- Normales hacia afuera ---
    _, nvec = tangent_and_normal(boundary0_u)
    n_out = orient_normals_outward(boundary0_u, nvec, mask1)

    # --- Distancia con signo del dominio FINAL ---
    sdf2 = signed_distance(mask2)

    # --- Desplazamiento u(s) por intersección de rayo normal ---
    u = normal_ray_intersection_u(boundary0_u, n_out, sdf2, tmax=P.TMAX_RAY, step=P.STEP_RAY)

    # --- Curva crecida reconstruida por u(s) ---
    grown = boundary0_u + n_out * u[:, None]

    # --- FFT de u(s) y reconstrucción truncada ---
    k, power = fft_spectrum(u)
    u_rec = reconstruct_from_fft(u, P.K_RECON)
    grown_rec = boundary0_u + n_out * u_rec[:, None]

    # ---------------------------------------------------------------------
    # Gráficas de chequeo
    # ---------------------------------------------------------------------

    # A) Máscaras inicial y final
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(mask1, origin='lower')
    plt.title("Máscara inicial")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask2, origin='lower')
    plt.title("Máscara final")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("A_masks.png", dpi=160)

    # B) Borde inicial + algunas normales
    plt.figure(figsize=(5, 5))
    plt.imshow(mask1, origin='lower')
    plt.plot(boundary0_u[:, 0], boundary0_u[:, 1], linewidth=1.2)
    idx = np.linspace(0, P.N_CURVE - 1, 80, dtype=int)
    for i in idx:
        p = boundary0_u[i]
        q = p + 8 * n_out[i]
        plt.plot([p[0], q[0]], [p[1], q[1]], linewidth=0.6)
    plt.title("Borde inicial + normales (muestra)")
    plt.axis('equal'); plt.axis('off')
    plt.tight_layout()
    plt.savefig("B_normals.png", dpi=160)

    # C) Borde coloreado por s (0→1)
    from matplotlib.collections import LineCollection
    from matplotlib import cm
    segments = np.stack([boundary0_u, np.roll(boundary0_u, -1, axis=0)], axis=1)
    lc = LineCollection(segments, array=s, linewidths=1.2, cmap=cm.get_cmap())
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.add_collection(lc)
    ax.set_xlim(0, P.SIZE); ax.set_ylim(0, P.SIZE)
    ax.set_aspect('equal'); ax.axis('off')
    plt.title("Borde coloreado por s")
    plt.tight_layout()
    plt.savefig("C_s_color.png", dpi=160)

    # D) Curva aplanada u(s)
    plt.figure(figsize=(7, 3))
    plt.plot(s, u, linewidth=1.5)
    plt.xlabel("s (fracción de longitud de arco)")
    plt.ylabel("u(s) (px)")
    plt.title("Desplazamiento normal u(s)")
    plt.tight_layout()
    plt.savefig("D_u_flat.png", dpi=160)

    # E) Espectro de potencia de u(s)
    plt.figure(figsize=(6, 3))
    plt.plot(k, power, linewidth=1.2)
    plt.xlabel("modo k (ciclos por vuelta)")
    plt.ylabel("potencia |U_k|^2")
    plt.title("Espectro de Fourier de u(s)")
    plt.tight_layout()
    plt.savefig("E_spectrum.png", dpi=160)

    # F) Overlay: borde final real vs reconstrucción con K modos
    plt.figure(figsize=(5, 5))
    plt.plot(boundary0_u[:, 0], boundary0_u[:, 1], linewidth=1.1, label="Inicial")
    plt.plot(grown[:, 0], grown[:, 1], linewidth=1.1, label="Final (rayos)")
    plt.plot(grown_rec[:, 0], grown_rec[:, 1], linewidth=1.1, label=f"Recon (K={P.K_RECON})")
    plt.axis('equal'); plt.axis('off')
    plt.legend()
    plt.title("Bordes: inicial / final / reconstrucción")
    plt.tight_layout()
    plt.savefig("F_overlay.png", dpi=160)

    # G) Fajas (opcional: GIF si hay ffmpeg)
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    ax.axis('off')
    # borde inicial una sola vez
    ax.contour(edge_initial.astype(float), levels=[0.5])
    im = ax.imshow(np.zeros_like(mask1), vmin=0, vmax=len(layers), interpolation='nearest')

    canvas = np.zeros_like(mask1, dtype=np.int32)
    def init():
        im.set_data(canvas); return [im]
    def update(frame):
        L = layers[frame]
        canvas[L] = frame + 1
        im.set_data(canvas); return [im]
    ani = FuncAnimation(fig, update, init_func=init, frames=len(layers), interval=200, blit=True)
    try:
        ani.save("G_fajas.gif", writer="ffmpeg", fps=6)
    except Exception as e:
        print("No se pudo guardar el GIF (¿ffmpeg instalado?):", e)
    plt.close(fig)

    print("Listo. Se guardaron imágenes A..F y (si se pudo) G_fajas.gif.")

if __name__ == "__main__":
    main()




