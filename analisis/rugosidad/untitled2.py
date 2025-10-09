# -*- coding: utf-8 -*-
"""
Autocorrelación de u_K(s) vs K_base — primer máximo (lag=0), plots mejorados
-----------------------------------------------------------------------------
- Reparametriza el contorno por arco.
- Curva base por FFT ±K; u_K(s) por normales (FFT) + ray casting.
- Autocorrelación circular NO normalizada (C_raw) y primer máximo: C_raw[0].
- Plots:
    (1) Autocorrelación C_raw(r) centrada en r=0 (simétrica), no normalizada.
    (2) Resumen C_raw(0) vs K en escala log–log.

Requisitos: numpy, matplotlib, scikit-image
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from skimage.measure import find_contours

# ================== Parámetros ==================
BASE_DIR = r"C:\Users\Marina\Documents\Labo 6\LuCam-app\analisis\rugosidad"
BASENAME = "Bin-P8139-190Oe-30ms-5Tw-"
IDX      = 99
EXTS     = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]

# Muestreo y Fourier
Msamples = 512
K_LIST   = list(range(2, 21, 2))   # barrido de armónicos

# Escala (para eje de autocorrelación en µm; poné 1.0 si preferís px)
PX_TO_UM = 0.4

# Qué K mostrar en el plot de autocorrelación (no normalizada)
SUBSET_K_FOR_AUTOCORR = [4, 12, 24]

# ================== IO ==================
def buscar_imagen(base_dir, basename, idx, exts):
    for ext in exts:
        p = os.path.join(base_dir, f"{basename}{idx}{ext}")
        if os.path.exists(p): return p
    cand = glob.glob(os.path.join(base_dir, f"{basename}{idx}*"))
    cand = [p for p in cand if os.path.splitext(p)[1].lower() in exts]
    return cand[0] if cand else ""

def cargar_binaria(path, thresh=0.5):
    im = io.imread(path)
    if im.ndim == 3 and im.shape[-1] == 4:
        im = im[..., :3]
    if im.ndim == 3:
        im = color.rgb2gray(im)
        return (im > thresh).astype(bool)
    if np.issubdtype(im.dtype, np.floating):
        return (im > thresh).astype(bool)
    elif np.issubdtype(im.dtype, np.integer):
        return (im > (np.iinfo(im.dtype).max * thresh)).astype(bool)
    else:
        return (im > thresh).astype(bool)

# ================== Curva y FFT ==================
def ensure_closed_no_dup(C):
    if np.allclose(C[0], C[-1]):
        return C[:-1].copy()
    return C.copy()

def resample_by_arclength(contour_yx, N):
    C = ensure_closed_no_dup(contour_yx)
    d = np.linalg.norm(np.diff(C, axis=0, append=C[:1]), axis=1)
    s = np.concatenate(([0.0], np.cumsum(d)))
    L = s[-1]
    st = np.linspace(0, L, N, endpoint=False)
    y = np.interp(st, s, np.r_[C[:,0], C[0,0]])
    x = np.interp(st, s, np.r_[C[:,1], C[0,1]])
    return np.column_stack([y, x])

def fft_lowpass_closed_equal_samples(curve_yx, K_keep):
    z = curve_yx[:,0] + 1j*curve_yx[:,1]
    Z = np.fft.fft(z)
    N = len(z)
    keep = np.zeros(N, dtype=bool)
    keep[0] = True
    K_keep = int(K_keep)
    for k in range(1, min(K_keep, N//2) + 1):
        keep[k % N]  = True
        keep[-k % N] = True
    z_s = np.fft.ifft(np.where(keep, Z, 0))
    return np.column_stack([z_s.real, z_s.imag])

# ================== Normales (FFT) ==================
def normals_fft_from_curve(y_ref, x_ref):
    z = y_ref + 1j*x_ref
    M = z.size
    Z = np.fft.fft(z)
    freqs = np.fft.fftfreq(M, d=1.0/M)  # enteros k
    dz_dt = np.fft.ifft(Z * (2j*np.pi*freqs))
    T = dz_dt / (np.abs(dz_dt) + 1e-15)
    N = 1j * T
    return np.column_stack([N.real, N.imag])

# ================== Intersecciones rayo–polilínea ==================
def ray_segment_intersection_one_side(p_yx, n_yx, Y, X):
    t_best = np.inf
    q_best = p_yx.copy()
    P = np.column_stack([Y, X])
    P = ensure_closed_no_dup(P)
    for i in range(len(P)):
        a = P[i]
        b = P[(i+1) % len(P)]
        ab = b - a
        A = np.column_stack([n_yx, -ab])
        rhs = a - p_yx
        det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
        if abs(det) < 1e-12:
            continue
        invA = (1.0/det) * np.array([[ A[1,1], -A[0,1]],
                                     [-A[1,0],  A[0,0]]])
        t, u = invA @ rhs
        if t >= 0.0 and -1e-12 <= u <= 1+1e-12:
            if t < t_best:
                t_best = t
                q_best = p_yx + t*n_yx
    return t_best, q_best

# ================== u(s) por normales FFT + ray casting ==================
def u_by_fft_normals_and_rays(y_ref, x_ref, y_real, x_real):
    Nrm = normals_fft_from_curve(y_ref, x_ref)
    u  = np.zeros_like(y_ref, float)
    qy = np.zeros_like(y_ref, float)
    qx = np.zeros_like(x_ref, float)
    for j in range(len(y_ref)):
        p = np.array([y_ref[j], x_ref[j]])
        n = Nrm[j]; n /= (np.linalg.norm(n) + 1e-15)
        tpos, qpos = ray_segment_intersection_one_side(p,  n, y_real, x_real)
        tneg, qneg = ray_segment_intersection_one_side(p, -n, y_real, x_real)
        cand = []
        if np.isfinite(tpos): cand.append((+tpos, qpos))
        if np.isfinite(tneg): cand.append((-tneg, qneg))
        if cand:
            cand.sort(key=lambda c: abs(c[0]))
            u[j]  = cand[0][0]
            qy[j], qx[j] = cand[0][1]
        else:
            u[j]  = 0.0
            qy[j], qx[j] = p
    return u, qy, qx

# ================== Autocorrelación circular ==================
def autocorr_circular(u):
    """
    C_raw[m] = <(u_j - ū) (u_{j+m} - ū)>_j  (no normalizada, circular)
    Se fuerza simetría: C[m] = C[-m] promediando con el reflejo circular.
    """
    u = np.asarray(u, float)
    u0 = u - np.mean(u)
    U = np.fft.fft(u0)
    C = np.fft.ifft(U * np.conj(U)).real / u0.size  # C_raw (circular)
    # --- simetrización exacta ---
    N = C.size
    idx_neg = (-np.arange(N)) % N
    C_sym = 0.5 * (C + C[idx_neg])
    return C_sym  # C_raw simetrizada


def centered_lag_axis(N, ds):
    """
    Lags enteros alineados con fftshift:
    [-N//2, ..., -1, 0, 1, ..., N//2-1] * ds   (largo N)
    """
    m = np.arange(N)
    m_centered = (m + N//2) % N - N//2
    return m_centered * ds, m_centered

def find_local_peaks(y):
    """Índices i con máximos locales estrictos (y[i-1] < y[i] >= y[i+1]) en 1..N-2.
    No usa SciPy. Devuelve lista ordenada."""
    idx = []
    N = len(y)
    for i in range(1, N-1):
        if y[i] > y[i-1] and y[i] >= y[i+1]:
            idx.append(i)
    return idx


# ================== MAIN ==================
if __name__ == "__main__":
    print("[inicio] Autocorrelación de u_K(s) vs K_base — primer máximo y plots limpios")

    # 1) cargar imagen y contorno
    path = buscar_imagen(BASE_DIR, BASENAME, IDX, EXTS)
    assert path, "No se encontró la imagen. Revisá BASE_DIR/BASENAME/IDX/EXTS."
    print(f"[input] {path}")

    im = cargar_binaria(path)
    C_all = find_contours(im, level=0.5)
    assert len(C_all) > 0, "No se encontró contorno."
    contour = ensure_closed_no_dup(max(C_all, key=len))  # (y,x)
    Y, X = contour[:,0], contour[:,1]

    # Reparametrización por arco
    Ceq = resample_by_arclength(contour, Msamples)  # (y,x)
    print(f"[ok] Contorno remuestreado: {Msamples} puntos")

    # Geometría básica
    ds_arr  = np.linalg.norm(np.diff(Ceq, axis=0, append=Ceq[:1]), axis=1)
    P_perim = float(np.sum(ds_arr))
    ds_mean = P_perim / Msamples
    print(f"[geom] Perímetro: {P_perim:.2f} px ({P_perim*PX_TO_UM:.2f} µm)")

    # 2) Barrido de K
    results = []
    print(f"[loop] Barrido K: {K_LIST}")
    for K in K_LIST:
        print(f"  • K={K}: FFT ±K y u(s) por normales…", end="", flush=True)
        smooth = fft_lowpass_closed_equal_samples(Ceq, K)
        yS, xS = smooth[:,0], smooth[:,1]

        uK_px, _qy, _qx = u_by_fft_normals_and_rays(yS, xS, Y, X)

        # Autocorrelación circular NO normalizada
        C_raw = autocorr_circular(uK_px)

        # Primer máximo = lag 0
        C0 = float(C_raw[0])

        results.append(dict(K=K, C_raw=C_raw, C0=C0))
        print(" listo.")

    # ================== PLOTS ==================
    # (1) Autocorrelación NO normalizada, centrada en r=0 y simétrica (algunos K)
    subset = [k for k in SUBSET_K_FOR_AUTOCORR if k in K_LIST]
    if subset:
        plt.figure(figsize=(11.5, 4.8))
        for K in subset:
            R = next(r for r in results if r["K"] == K)
            C_raw = R["C_raw"]                      # ya simetrizada
            Cc = np.fft.fftshift(C_raw)            # centrar pico en r=0
            r_axis_px, mcent = centered_lag_axis(len(C_raw), ds_mean)
            r_axis = r_axis_px * PX_TO_UM
    
            # índice de r=0 en el vector "shifted"
            zero_idx = np.where(mcent == 0)[0][0]
    
            # detectar picos locales (sobre el array centrado)
            peaks = find_local_peaks(Cc)
    
            # separar picos a la izquierda (<0) y derecha (>0) del cero
            left_peaks  = [i for i in peaks if i <  zero_idx]
            right_peaks = [i for i in peaks if i >  zero_idx]
    
            # elegir el pico lateral más cercano a 0 de cada lado (si existe)
            left_star  = max(left_peaks)  if left_peaks  else None
            right_star = min(right_peaks) if right_peaks else None
    
            # -- curva
            plt.plot(r_axis, Cc, lw=1.25, label=f"K={K}")
    
            # -- marcar el pico principal (r=0), que es C_raw[0]
            plt.scatter([r_axis[zero_idx]], [Cc[zero_idx]], s=36, zorder=5, edgecolors='k')
            plt.annotate(f"C(0)={Cc[zero_idx]:.2f}",
                         xy=(r_axis[zero_idx], Cc[zero_idx]),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=0.5))
    
            # -- marcar los primeros picos laterales simétricos (si existen)
            for star_idx in [left_star, right_star]:
                if star_idx is not None:
                    plt.scatter([r_axis[star_idx]], [Cc[star_idx]], s=24, zorder=5)
                    plt.annotate(f"{r_axis[star_idx]:.2f} µm",
                                 xy=(r_axis[star_idx], Cc[star_idx]),
                                 xytext=(0, -14), textcoords='offset points',
                                 ha='center', fontsize=8)
    
        plt.axvline(0.0, color='k', lw=1.0, ls='--', alpha=0.6)
        plt.xlabel('lag r [µm]')
        plt.ylabel('C_raw(r) [px²]')
        plt.title('Autocorrelación circular NO normalizada (centrada y con picos marcados)')
        plt.legend(ncol=4, fontsize=9)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    # (2) Resumen: C_raw(0) vs K en log–log
    K_arr  = np.array([r["K"] for r in results], int)
    C0_arr = np.array([r["C0"] for r in results], float)

    # Fit lineal en log–log (solo puntos positivos)
    mask_pos = (K_arr > 0) & (C0_arr > 0)
    xlog = np.log10(K_arr[mask_pos])
    ylog = np.log10(C0_arr[mask_pos])
    m, b = np.polyfit(xlog, ylog, 1) if len(xlog) >= 2 else (np.nan, np.nan)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.loglog(K_arr, C0_arr, 'o', ms=5, label='C_raw(0) datos')
    if np.isfinite(m):
        xfit = np.linspace(xlog.min(), xlog.max(), 200)
        yfit = m * xfit + b
        ax.plot(10**xfit, 10**yfit, '-', lw=1.5, label=f"fit: slope = {m:.3f}")
    ax.set_xlabel("K (modos mantenidos)")
    ax.set_ylabel("C_raw(0) = var(u_K) [px²]")
    ax.set_title("Primer máximo de autocorrelación vs K (log–log)")
    ax.grid(which='both', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"[fin] Ajuste log–log: pendiente ~ {m:.4f}  (si se muestran NaN, faltan puntos válidos)")
