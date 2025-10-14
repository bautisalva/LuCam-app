"""
-----------------------------------------------------------------------------------
Entradas principales:
  - mask: ndarray booleana (True = interior del dominio) ya binarizada
  - K:    entero (modos ±K retenidos al suavizar la curva de referencia)

Salidas principales (en un diccionario):
  - r_px, B_px:  desplazamientos y B(r) en píxeles
  - r_um, B_um:  lo mismo escalado a µm (según px_to_um)
  - q_um_inv, S: frecuencias espaciales (1/µm) y espectro S(q)
  - extras: contornos y u(s) por si se quieren reutilizar en otros pasos

"""
from __future__ import annotations
import numpy as np
from skimage import measure
from typing import Tuple, Dict, Any

# ================== Utilidades de curva cerrada ==================

def _ensure_closed_no_dup(C: np.ndarray) -> np.ndarray:
    """Asegura cierre y evita duplicar el primer punto al final."""
    if np.allclose(C[0], C[-1]):
        return C[:-1].copy()
    return C.copy()


def _perimeter(C: np.ndarray) -> float:
    d = np.linalg.norm(np.diff(np.vstack([C, C[0]]), axis=0), axis=1)
    return float(np.sum(d))


def _resample_by_arclength(contour_yx: np.ndarray, N: int) -> np.ndarray:
    """Reparametriza contorno (y,x) a N puntos equiespaciados en arclonga."""
    C = _ensure_closed_no_dup(contour_yx)
    d = np.linalg.norm(np.diff(np.vstack([C, C[0]]), axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(d)))
    L = s[-1]
    st = np.linspace(0, L, N, endpoint=False)
    y = np.interp(st, s, np.r_[C[:, 0], C[0, 0]])
    x = np.interp(st, s, np.r_[C[:, 1], C[0, 1]])
    return np.column_stack([y, x])


def _fft_lowpass_closed_equal_samples(curve_yx: np.ndarray, K_keep: int) -> np.ndarray:
    """Low-pass FFT (±K_keep) sobre z=y+i x. Devuelve (y,x) suave."""
    z = curve_yx[:, 0] + 1j * curve_yx[:, 1]
    Z = np.fft.fft(z)
    N = len(z)
    keep = np.zeros(N, dtype=bool)
    keep[0] = True
    kmax = min(max(int(K_keep), 0), N // 2)
    for k in range(1, kmax + 1):
        keep[k] = True
        keep[-k] = True
    z_s = np.fft.ifft(np.where(keep, Z, 0))
    return np.column_stack([z_s.real, z_s.imag])


def _normals_fft_from_curve(y_ref: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """Normales unitarias N = i * T̂, con T̂=(dz/dt)/|dz/dt| y z=y+i x."""
    z = y_ref + 1j * x_ref
    M = z.size
    Z = np.fft.fft(z)
    freqs = np.fft.fftfreq(M, d=1.0 / M)  # enteros k
    dZ = Z * (2j * np.pi * freqs)
    dz_dt = np.fft.ifft(dZ)
    T = dz_dt / (np.abs(dz_dt) + 1e-15)
    Nrm = 1j * T
    return np.column_stack([Nrm.real, Nrm.imag])  # (ny, nx)

# ================== Intersecciones rayo–polilínea ==================

def _ray_segment_intersection_one_side(p_yx: np.ndarray, n_yx: np.ndarray, Y: np.ndarray, X: np.ndarray) -> Tuple[float, np.ndarray]:
    """Intersección de rayo p + t n (t>=0) con polilínea (Y,X) cerrada."""
    t_best = np.inf
    q_best = p_yx.copy()
    P = np.column_stack([Y, X])
    P = _ensure_closed_no_dup(P)
    for i in range(len(P)):
        a = P[i]
        b = P[(i + 1) % len(P)]
        ab = b - a
        A = np.column_stack([n_yx, -ab])
        rhs = a - p_yx
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            continue
        invA = (1.0 / det) * np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]])
        t, u = invA @ rhs
        if t >= 0.0 and -1e-12 <= u <= 1 + 1e-12:
            if t < t_best:
                t_best = t
                q_best = p_yx + t * n_yx
    return t_best, q_best


def _u_by_fft_normals_and_rays(y_ref: np.ndarray, x_ref: np.ndarray, y_real: np.ndarray, x_real: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """u por intersección del rayo ±normal FFT con contorno real."""
    Nrm = _normals_fft_from_curve(y_ref, x_ref)
    u = np.zeros_like(y_ref, float)
    qy = np.zeros_like(y_ref, float)
    qx = np.zeros_like(x_ref, float)
    for j in range(len(y_ref)):
        p = np.array([y_ref[j], x_ref[j]])
        n = Nrm[j]
        n /= (np.linalg.norm(n) + 1e-15)
        tpos, qpos = _ray_segment_intersection_one_side(p, n, y_real, x_real)
        tneg, qneg = _ray_segment_intersection_one_side(p, -n, y_real, x_real)
        cand = []
        if np.isfinite(tpos):
            cand.append((+tpos, qpos))
        if np.isfinite(tneg):
            cand.append((-tneg, qneg))
        if cand:
            cand.sort(key=lambda c: abs(c[0]))
            u[j] = cand[0][0]
            qy[j], qx[j] = cand[0][1]
        else:
            u[j] = 0.0
            qy[j], qx[j] = p
    return u, qy, qx

# ================== B(r) denso y S(q) por DFT ==================

def _circular_shift_frac(u: np.ndarray, shift_samples: float) -> np.ndarray:
    """Corrimiento fraccional periódico estable (interpolación lineal)."""
    u = np.asarray(u)
    N = u.size
    i = np.arange(N, dtype=np.float64)
    x = np.mod(i - shift_samples, N)
    x = np.where(x >= N - 1e-12, x - N, x)
    i0 = np.floor(x).astype(np.int64)
    a = x - i0
    i1 = (i0 + 1) % N
    return (1.0 - a) * u[i0] + a * u[i1]


def _B_of_r_dense(u_px: np.ndarray, P_perimeter_px: float, r_max_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """B(r) = <[u(z+r)-u(z)]^2> con r casi continuo hasta r_max_frac·P."""
    u0 = u_px - np.mean(u_px)
    N = len(u0)
    ds = P_perimeter_px / N
    r_max = r_max_frac * P_perimeter_px
    n_max = int(np.floor(r_max / ds))
    r_vals = ds * np.arange(1, n_max + 1)
    B = np.empty_like(r_vals, dtype=float)
    for i, r in enumerate(r_vals):
        shift = r / ds
        u_shift = _circular_shift_frac(u0, shift)
        dif = u_shift - u0
        B[i] = np.mean(dif * dif)
    return r_vals, B


def _Sq_from_DFT(u_px: np.ndarray, P_perimeter_px: float, px_to_um: float, use_hann: bool) -> Tuple[np.ndarray, np.ndarray]:
    """S(q_n) = |(1/N) Σ u_j e^{-2π i n j/N}|^2 con q_n = n/L (ciclos/µm)."""
    u0 = u_px - np.mean(u_px)
    N = len(u0)
    if use_hann:
        w = np.hanning(N)
        u0 = u0 * w
        norm = (np.sum(w) ** 2)
    else:
        norm = N ** 2
    U = np.fft.fft(u0) / N
    S = (np.abs(U) ** 2) * (N ** 2) / norm
    n = np.fft.fftfreq(N, d=1.0 / N)
    mask = (n > 0)
    n = n[mask].astype(int)
    S = S[mask]
    L_um = P_perimeter_px * px_to_um
    q_um_inv = n / L_um
    return q_um_inv, S

# ================== Clase principal ==================

class RoughnessBrSq:
    def __init__(self, Msamples: int = 1024, px_to_um: float = 1.0,
                 r_max_frac: float = 0.8, use_hann_for_S: bool = True):
        """
        Msamples: puntos con los que parametrizamos la curva por arclonga.
        px_to_um: escala para pasar de px a µm (1.0 = dejar en px).
        r_max_frac: r_max como fracción del perímetro para B(r).
        use_hann_for_S: si True, aplica ventana de Hann antes de la DFT.
        """
        self.M = int(Msamples)
        self.px_to_um = float(px_to_um)
        self.r_max_frac = float(r_max_frac)
        self.use_hann_for_S = bool(use_hann_for_S)

    # ---- Paso 1: contorno principal (y,x) desde máscara booleana ----
    @staticmethod
    def _largest_contour_yx(mask: np.ndarray) -> np.ndarray:
        conts = measure.find_contours(mask.astype(float), level=0.5)
        if not conts:
            raise ValueError("No se encontró contorno en la máscara.")
        contour = max(conts, key=len)  # (y,x)
        return _ensure_closed_no_dup(contour)

    def analyze(self, mask: np.ndarray, K: int) -> Dict[str, Any]:
        """
        Calcula B(r) y S(q) para la máscara binaria dada y un K base.
        Retorna un dict con:
          r_px, B_px, r_um, B_um, q_um_inv, S,
          y extras: u_px, s_px, P_px, yS/xS (curva suave), Y/X (real).
        """
        if mask.dtype != bool:
            raise TypeError("'mask' debe ser booleana (True = interior). Ya binarizada.")
        if int(K) < 0:
            raise ValueError("K debe ser un entero no negativo.")

        # Contorno real y remuestreo por arclonga
        contour = self._largest_contour_yx(mask)  # (y,x)
        Ceq = _resample_by_arclength(contour, self.M)  # (y,x)

        # Curva base: suavizado Fourier con ±K
        smooth = _fft_lowpass_closed_equal_samples(Ceq, int(K))  # (y,x)
        yS, xS = smooth[:, 0], smooth[:, 1]

        # Normal FFT + "ray casting" al contorno real para obtener u(s)
        Y, X = contour[:, 0], contour[:, 1]
        u_px, qy, qx = _u_by_fft_normals_and_rays(yS, xS, Y, X)

        # Eje s y perímetro de la curva suave (la referencia geométrica)
        ds = np.linalg.norm(np.diff(np.vstack([smooth, smooth[0]]), axis=0), axis=1)
        s_px = np.concatenate(([0.0], np.cumsum(ds[:-1])))
        P_px = float(np.sum(ds))

        # B(r) (px y µm)
        r_px, B_px = _B_of_r_dense(u_px, P_px, self.r_max_frac)
        r_um = r_px * self.px_to_um
        B_um = B_px * (self.px_to_um ** 2)

        # S(q)
        q_um_inv, S = _Sq_from_DFT(u_px, P_px, self.px_to_um, self.use_hann_for_S)

        return dict(
            r_px=r_px, B_px=B_px,
            r_um=r_um, B_um=B_um,
            q_um_inv=q_um_inv, S=S,
            extras=dict(
                u_px=u_px, s_px=s_px, P_px=P_px,
                yS=yS, xS=xS, Y=Y, X=X, qy=qy, qx=qx
            )
        )

# ================== Ejemplo de uso rápido (smoke test) ==================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Máscara binaria sintética: disco con leve deformación (modo 6)
    H, W = 256, 256
    y, x = np.mgrid[:H, :W]
    cy, cx, r0 = H/2, W/2, 80
    base = (x - cx)**2 + (y - cy)**2 <= r0**2
    theta = np.arctan2(y - cy, x - cx)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = base | (r < r0 + 4*np.cos(6*theta))

    analyzer = RoughnessBrSq(Msamples=1024, px_to_um=0.4, r_max_frac=0.8, use_hann_for_S=True)
    out = analyzer.analyze(mask.astype(bool), K=4)

    # Plots mínimos solo para inspección (podés borrar estas líneas)
    plt.figure(); plt.loglog(out["r_um"], out["B_um"], '.');
    plt.xlabel('r [µm]'); plt.ylabel('B(r) [µm²]'); plt.title('B(r)')
    plt.figure(); plt.loglog(out["q_um_inv"], out["S"], '.');
    plt.xlabel('q [1/µm]'); plt.ylabel('S(q) [a.u.]'); plt.title('S(q)')
    plt.show()
# -*- coding: utf-8 -*-
"""
RoughnessBrSq — Clase minimal **solo** para B(r) y S(q) desde una imagen binaria + K
-----------------------------------------------------------------------------------
Entradas principales:
  - mask: ndarray booleana (True = interior del dominio) ya binarizada
  - K:    entero (modos ±K retenidos al suavizar la curva de referencia)

Salidas principales (en un diccionario):
  - r_px, B_px:  desplazamientos y B(r) en píxeles
  - r_um, B_um:  lo mismo escalado a µm (según px_to_um)
  - q_um_inv, S: frecuencias espaciales (1/µm) y espectro S(q)
  - extras: contornos y u(s) por si se quieren reutilizar en otros pasos

Sin ajustes ni exponente ζ (vos los hacés luego). La clase no plotea.
Depende de: numpy, scikit-image (measure.find_contours)
"""
from __future__ import annotations
import numpy as np
from skimage import measure
from typing import Tuple, Dict, Any

# ================== Utilidades de curva cerrada ==================

def _ensure_closed_no_dup(C: np.ndarray) -> np.ndarray:
    """Asegura cierre y evita duplicar el primer punto al final."""
    if np.allclose(C[0], C[-1]):
        return C[:-1].copy()
    return C.copy()


def _perimeter(C: np.ndarray) -> float:
    d = np.linalg.norm(np.diff(np.vstack([C, C[0]]), axis=0), axis=1)
    return float(np.sum(d))


def _resample_by_arclength(contour_yx: np.ndarray, N: int) -> np.ndarray:
    """Reparametriza contorno (y,x) a N puntos equiespaciados en arclonga."""
    C = _ensure_closed_no_dup(contour_yx)
    d = np.linalg.norm(np.diff(np.vstack([C, C[0]]), axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(d)))
    L = s[-1]
    st = np.linspace(0, L, N, endpoint=False)
    y = np.interp(st, s, np.r_[C[:, 0], C[0, 0]])
    x = np.interp(st, s, np.r_[C[:, 1], C[0, 1]])
    return np.column_stack([y, x])


def _fft_lowpass_closed_equal_samples(curve_yx: np.ndarray, K_keep: int) -> np.ndarray:
    """Low-pass FFT (±K_keep) sobre z=y+i x. Devuelve (y,x) suave."""
    z = curve_yx[:, 0] + 1j * curve_yx[:, 1]
    Z = np.fft.fft(z)
    N = len(z)
    keep = np.zeros(N, dtype=bool)
    keep[0] = True
    kmax = min(max(int(K_keep), 0), N // 2)
    for k in range(1, kmax + 1):
        keep[k] = True
        keep[-k] = True
    z_s = np.fft.ifft(np.where(keep, Z, 0))
    return np.column_stack([z_s.real, z_s.imag])


def _normals_fft_from_curve(y_ref: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """Normales unitarias N = i * T̂, con T̂=(dz/dt)/|dz/dt| y z=y+i x."""
    z = y_ref + 1j * x_ref
    M = z.size
    Z = np.fft.fft(z)
    freqs = np.fft.fftfreq(M, d=1.0 / M)  # enteros k
    dZ = Z * (2j * np.pi * freqs)
    dz_dt = np.fft.ifft(dZ)
    T = dz_dt / (np.abs(dz_dt) + 1e-15)
    Nrm = 1j * T
    return np.column_stack([Nrm.real, Nrm.imag])  # (ny, nx)

# ================== Intersecciones rayo–polilínea ==================

def _ray_segment_intersection_one_side(p_yx: np.ndarray, n_yx: np.ndarray, Y: np.ndarray, X: np.ndarray) -> Tuple[float, np.ndarray]:
    """Intersección de rayo p + t n (t>=0) con polilínea (Y,X) cerrada."""
    t_best = np.inf
    q_best = p_yx.copy()
    P = np.column_stack([Y, X])
    P = _ensure_closed_no_dup(P)
    for i in range(len(P)):
        a = P[i]
        b = P[(i + 1) % len(P)]
        ab = b - a
        A = np.column_stack([n_yx, -ab])
        rhs = a - p_yx
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            continue
        invA = (1.0 / det) * np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]])
        t, u = invA @ rhs
        if t >= 0.0 and -1e-12 <= u <= 1 + 1e-12:
            if t < t_best:
                t_best = t
                q_best = p_yx + t * n_yx
    return t_best, q_best


def _u_by_fft_normals_and_rays(y_ref: np.ndarray, x_ref: np.ndarray, y_real: np.ndarray, x_real: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """u por intersección del rayo ±normal FFT con contorno real."""
    Nrm = _normals_fft_from_curve(y_ref, x_ref)
    u = np.zeros_like(y_ref, float)
    qy = np.zeros_like(y_ref, float)
    qx = np.zeros_like(x_ref, float)
    for j in range(len(y_ref)):
        p = np.array([y_ref[j], x_ref[j]])
        n = Nrm[j]
        n /= (np.linalg.norm(n) + 1e-15)
        tpos, qpos = _ray_segment_intersection_one_side(p, n, y_real, x_real)
        tneg, qneg = _ray_segment_intersection_one_side(p, -n, y_real, x_real)
        cand = []
        if np.isfinite(tpos):
            cand.append((+tpos, qpos))
        if np.isfinite(tneg):
            cand.append((-tneg, qneg))
        if cand:
            cand.sort(key=lambda c: abs(c[0]))
            u[j] = cand[0][0]
            qy[j], qx[j] = cand[0][1]
        else:
            u[j] = 0.0
            qy[j], qx[j] = p
    return u, qy, qx

# ================== B(r) denso y S(q) por DFT ==================

def _circular_shift_frac(u: np.ndarray, shift_samples: float) -> np.ndarray:
    """Corrimiento fraccional periódico estable (interpolación lineal)."""
    u = np.asarray(u)
    N = u.size
    i = np.arange(N, dtype=np.float64)
    x = np.mod(i - shift_samples, N)
    x = np.where(x >= N - 1e-12, x - N, x)
    i0 = np.floor(x).astype(np.int64)
    a = x - i0
    i1 = (i0 + 1) % N
    return (1.0 - a) * u[i0] + a * u[i1]


def _B_of_r_dense(u_px: np.ndarray, P_perimeter_px: float, r_max_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """B(r) = <[u(z+r)-u(z)]^2> con r casi continuo hasta r_max_frac·P."""
    u0 = u_px - np.mean(u_px)
    N = len(u0)
    ds = P_perimeter_px / N
    r_max = r_max_frac * P_perimeter_px
    n_max = int(np.floor(r_max / ds))
    r_vals = ds * np.arange(1, n_max + 1)
    B = np.empty_like(r_vals, dtype=float)
    for i, r in enumerate(r_vals):
        shift = r / ds
        u_shift = _circular_shift_frac(u0, shift)
        dif = u_shift - u0
        B[i] = np.mean(dif * dif)
    return r_vals, B


def _Sq_from_DFT(u_px: np.ndarray, P_perimeter_px: float, px_to_um: float, use_hann: bool) -> Tuple[np.ndarray, np.ndarray]:
    """S(q_n) = |(1/N) Σ u_j e^{-2π i n j/N}|^2 con q_n = n/L (ciclos/µm)."""
    u0 = u_px - np.mean(u_px)
    N = len(u0)
    if use_hann:
        w = np.hanning(N)
        u0 = u0 * w
        norm = (np.sum(w) ** 2)
    else:
        norm = N ** 2
    U = np.fft.fft(u0) / N
    S = (np.abs(U) ** 2) * (N ** 2) / norm
    n = np.fft.fftfreq(N, d=1.0 / N)
    mask = (n > 0)
    n = n[mask].astype(int)
    S = S[mask]
    L_um = P_perimeter_px * px_to_um
    q_um_inv = n / L_um
    return q_um_inv, S

# ================== Clase principal ==================

class RoughnessBrSq:
    def __init__(self, Msamples: int = 1024, px_to_um: float = 1.0,
                 r_max_frac: float = 0.8, use_hann_for_S: bool = True):
        """
        Msamples: puntos con los que parametrizamos la curva por arclonga.
        px_to_um: escala para pasar de px a µm (1.0 = dejar en px).
        r_max_frac: r_max como fracción del perímetro para B(r).
        use_hann_for_S: si True, aplica ventana de Hann antes de la DFT.
        """
        self.M = int(Msamples)
        self.px_to_um = float(px_to_um)
        self.r_max_frac = float(r_max_frac)
        self.use_hann_for_S = bool(use_hann_for_S)

    # ---- Paso 1: contorno principal (y,x) desde máscara booleana ----
    @staticmethod
    def _largest_contour_yx(mask: np.ndarray) -> np.ndarray:
        conts = measure.find_contours(mask.astype(float), level=0.5)
        if not conts:
            raise ValueError("No se encontró contorno en la máscara.")
        contour = max(conts, key=len)  # (y,x)
        return _ensure_closed_no_dup(contour)

    def analyze(self, mask: np.ndarray, K: int) -> Dict[str, Any]:
        """
        Calcula B(r) y S(q) para la máscara binaria dada y un K base.
        Retorna un dict con:
          r_px, B_px, r_um, B_um, q_um_inv, S,
          y extras: u_px, s_px, P_px, yS/xS (curva suave), Y/X (real).
        """
        if mask.dtype != bool:
            raise TypeError("'mask' debe ser booleana (True = interior). Ya binarizada.")
        if int(K) < 0:
            raise ValueError("K debe ser un entero no negativo.")

        # Contorno real y remuestreo por arclonga
        contour = self._largest_contour_yx(mask)  # (y,x)
        Ceq = _resample_by_arclength(contour, self.M)  # (y,x)

        # Curva base: suavizado Fourier con ±K
        smooth = _fft_lowpass_closed_equal_samples(Ceq, int(K))  # (y,x)
        yS, xS = smooth[:, 0], smooth[:, 1]

        # Normal FFT + "ray casting" al contorno real para obtener u(s)
        Y, X = contour[:, 0], contour[:, 1]
        u_px, qy, qx = _u_by_fft_normals_and_rays(yS, xS, Y, X)

        # Eje s y perímetro de la curva suave (la referencia geométrica)
        ds = np.linalg.norm(np.diff(np.vstack([smooth, smooth[0]]), axis=0), axis=1)
        s_px = np.concatenate(([0.0], np.cumsum(ds[:-1])))
        P_px = float(np.sum(ds))

        # B(r) (px y µm)
        r_px, B_px = _B_of_r_dense(u_px, P_px, self.r_max_frac)
        r_um = r_px * self.px_to_um
        B_um = B_px * (self.px_to_um ** 2)

        # S(q)
        q_um_inv, S = _Sq_from_DFT(u_px, P_px, self.px_to_um, self.use_hann_for_S)

        return dict(
            r_px=r_px, B_px=B_px,
            r_um=r_um, B_um=B_um,
            q_um_inv=q_um_inv, S=S,
            extras=dict(
                u_px=u_px, s_px=s_px, P_px=P_px,
                yS=yS, xS=xS, Y=Y, X=X, qy=qy, qx=qx
            )
        )

