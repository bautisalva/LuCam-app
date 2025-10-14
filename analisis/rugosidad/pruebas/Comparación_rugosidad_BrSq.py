import os, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from skimage.measure import find_contours

# ================== Parámetros ==================
BASE_DIR = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe\080"
BASENAME = "Bin-P8137-080Oe-100ms-"
IDX      = 5
EXTS     = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]

# Muestreo y Fourier
Msamples = 1024          # densidad para parametrizar el contorno
K_LIST   = [1]    # armónicos para mi método (curva suave)

# B(r) y S(q)
R_MAX_FRAC     = 0.80       # r_max como fracción del perímetro
B_FIT_LOG10    = (-1, 0.6) # ventana de ajuste para B en log10(r)
S_FIT_LOG10    = (0.1, 0.8) # ventana de ajuste para S en log10(q) (se usará sólo para filtrar; el fit se hace en lineal)
USE_HANN_FOR_S = True      # ventana de Hann para S(q) (False = sin ventana)
QT_HIGH_FRAC   = 1       # recorte visual de altas frecuencias (opcional, sólo para plot)

# Escala
PX_TO_UM = 0.4             # µm/px

# ================== Utilidades generales ==================
LN10 = np.log(10.0)
def to_ln_window(win_log10):
    return (win_log10[0]*LN10, win_log10[1]*LN10)

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
    if im.ndim == 3 and im.shape[-1] == 4: im = im[..., :3]
    if im.ndim == 3:
        im = color.rgb2gray(im); return (im > thresh).astype(bool)
    if np.issubdtype(im.dtype, np.floating):
        return (im > thresh).astype(bool)
    elif np.issubdtype(im.dtype, np.integer):
        return (im > (np.iinfo(im.dtype).max * thresh)).astype(bool)
    else:
        return (im > thresh).astype(bool)

# ================== Contorno y FFT de la curva ==================
def ensure_closed_no_dup(C):
    if np.allclose(C[0], C[-1]): return C[:-1].copy()
    return C.copy()

def resample_by_arclength(contour_yx, N):
    """Reparametriza contorno (y,x) a N puntos equiespaciados por arco."""
    C = ensure_closed_no_dup(contour_yx)
    d = np.linalg.norm(np.diff(C, axis=0, append=C[:1]), axis=1)
    s = np.concatenate(([0.0], np.cumsum(d))); L = s[-1]
    st = np.linspace(0, L, N, endpoint=False)
    y = np.interp(st, s, np.r_[C[:,0], C[0,0]])
    x = np.interp(st, s, np.r_[C[:,1], C[0,1]])
    return np.column_stack([y, x])

def fft_lowpass_closed_equal_samples(curve_yx, K_keep):
    """Low-pass FFT (±K_keep) sobre z=y+i x. Devuelve (y,x) suave."""
    z = curve_yx[:,0] + 1j*curve_yx[:,1]
    Z = np.fft.fft(z); N = len(z)
    keep = np.zeros(N, dtype=bool); keep[0] = True
    for k in range(1, K_keep+1):
        keep[k % N] = True; keep[-k % N] = True
    z_s = np.fft.ifft(np.where(keep, Z, 0))
    return np.column_stack([z_s.real, z_s.imag])

def normals_fft_from_curve(y_ref, x_ref):
    """Normales unitarias N = i * T̂, con T̂=(dz/dt)/|dz/dt| y z=y+i x."""
    z = y_ref + 1j*x_ref
    M = z.size
    Z = np.fft.fft(z)
    freqs = np.fft.fftfreq(M, d=1.0/M)  # enteros k
    dZ = Z * (2j*np.pi*freqs)
    dz_dt = np.fft.ifft(dZ)
    T = dz_dt / (np.abs(dz_dt) + 1e-15)
    N = 1j * T
    return np.column_stack([N.real, N.imag])  # (ny, nx)

# ================== Intersecciones rayo–polilínea ==================
def ray_segment_intersection_one_side(p_yx, n_yx, Y, X):
    """Intersección de rayo p + t n (t>=0) con polilínea (Y,X) cerrada."""
    t_best = np.inf
    q_best = p_yx.copy()
    P = np.column_stack([Y, X])
    P = ensure_closed_no_dup(P)
    for i in range(len(P)):
        a = P[i]; b = P[(i+1) % len(P)]
        ab = b - a
        A = np.column_stack([n_yx, -ab])
        rhs = a - p_yx
        det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
        if abs(det) < 1e-12: continue
        invA = (1.0/det) * np.array([[ A[1,1], -A[0,1]],
                                     [-A[1,0],  A[0,0]]])
        t, u = invA @ rhs
        if t >= 0.0 and -1e-12 <= u <= 1+1e-12:
            if t < t_best:
                t_best = t
                q_best = p_yx + t*n_yx
    return t_best, q_best

# ================== u(s): Mi método (normales FFT) ==================
def u_by_fft_normals_and_rays(y_ref, x_ref, y_real, x_real):
    """u por intersección del rayo ±normal FFT con contorno real."""
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

# ================== u(θ): Pablo = círculo de radio medio ==================
def centroid_from_polygon(Y, X):
    """Centroide poligonal (y,x) con fórmula de área con signo."""
    x = X; y = Y
    x2 = np.roll(x, -1); y2 = np.roll(y, -1)
    a = x*y2 - x2*y
    A = 0.5*np.sum(a)
    if abs(A) < 1e-12:
        return np.mean(Y), np.mean(X)
    cx = (1/(6*A)) * np.sum((x + x2) * a)
    cy = (1/(6*A)) * np.sum((y + y2) * a)
    return cy, cx  # (y,x)

def pablo_circle_mean_radius(Y, X, Msamples):
    """
    Centro (yc,xc) = centroide; r(θ) = intersección radial; r0 = <r>; u(θ)=r(θ)-r0.
    Devuelve (y_ref,x_ref) del círculo r0, u(θ) [px], s y perímetro P.
    """
    yc, xc = centroid_from_polygon(Y, X)
    thetas = np.linspace(0, 2*np.pi, Msamples, endpoint=False)
    r_theta = np.zeros(Msamples, float)
    for m, th in enumerate(thetas):
        n = np.array([np.sin(th), np.cos(th)])  # (ny, nx)
        t, _ = ray_segment_intersection_one_side(np.array([yc, xc]), n, Y, X)
        r_theta[m] = t if np.isfinite(t) else 0.0
    r0 = np.mean(r_theta)
    y_ref = yc + r0*np.sin(thetas)
    x_ref = xc + r0*np.cos(thetas)
    u_px  = r_theta - r0
    P     = 2*np.pi*r0
    s     = thetas * r0
    return y_ref, x_ref, u_px, s, P, (yc, xc, r0, thetas, r_theta)

# ================== B(r) “lo más continuo posible” ==================
# === PARCHE: reemplaza la función _circular_shift_frac por una versión estable ===
def _circular_shift_frac(u, shift_samples):
    """
    Desplaza u fraccionalmente (periódico) en 'shift_samples' muestras.
    Maneja con cuidado el borde x≈N para evitar i0==N.
    """
    u = np.asarray(u)
    N = u.size

    # Coordenadas base (float para evitar overflow en mod)
    i = np.arange(N, dtype=np.float64)

    # x en [0, N) con protección numérica
    x = np.mod(i - shift_samples, N)          # teórico: [0, N)
    # si por redondeo cae en N, lo traemos a [0, N)
    x = np.where(x >= N - 1e-12, x - N, x)    # ahora garantizado x ∈ [0, N)

    # indices entero + peso fraccional
    i0 = np.floor(x).astype(np.int64)
    a  = x - i0                                # a ∈ [0,1)
    i1 = (i0 + 1) % N

    return (1.0 - a) * u[i0] + a * u[i1]


def B_of_r_dense(u_px, P_perimeter_px, r_max_frac=0.5):
    """
    B(r) = <[u(z+r)-u(z)]^2>_z evaluado de forma densa:
    - r barre TODOS los desplazamientos posibles uniformemente desde ~ds hasta r_max.
    - Usa corrimiento fraccional para evitar aliasing (casi continuo).
    Devuelve r_px (N//2 valores) y B_px2.
    """
    u0 = u_px - np.mean(u_px)
    N  = len(u0)
    ds = P_perimeter_px / N
    # usamos tantos r como N//2 (desde 1*ds hasta r_max)
    r_max = r_max_frac * P_perimeter_px
    n_max = int(np.floor(r_max / ds))
    r_vals = ds * np.arange(1, n_max+1)  # 1*ds, 2*ds, ..., n_max*ds
    B = np.empty_like(r_vals, dtype=float)
    for i, r in enumerate(r_vals):
        shift = r/ds
        u_shift = _circular_shift_frac(u0, shift)
        dif = u_shift - u0
        B[i] = np.mean(dif*dif)
    return r_vals, B

# ================== S(q) por DFT y FIT en dominio lineal ==================
def Sq_from_DFT(u_px, P_perimeter_px, px_to_um=1.0, use_hann=False):
    """
    S(q_n) = | (1/N) sum_j u_j e^{-2π i n j/N} |^2   con q_n = n/L_um (ciclos/µm).
    Devuelve q>0 y S(q).
    """
    u0 = u_px - np.mean(u_px)
    N  = len(u0)
    if use_hann:
        w = np.hanning(N); u0 = u0*w; norm = (np.sum(w)**2)
    else:
        norm = N**2
    U = np.fft.fft(u0)/N
    S = (np.abs(U)**2) * (N**2) / norm  # corrige si hay ventana

    n = np.fft.fftfreq(N, d=1.0/N)  # enteros (...,-2,-1,0,1,2,...)
    mask = (n > 0)
    n = n[mask].astype(int)
    S = S[mask]
    L_um = P_perimeter_px * px_to_um
    q_um_inv = n / L_um  # [1/µm]
    return q_um_inv, S

def fit_S_powerlaw_linear(q_um_inv, S, fit_window_log10=(0.0, 0.8), q0=1.0):
    """
    Ajusta S(q) ≈ S0 * (q/q0)^(-(1+2ζ)) en DOMINIO LINEAL (sin log):
    - Para un α = (1+2ζ), el mejor S0 (LS) es: S0 = (∑ w_i y_i x_i) / (∑ w_i x_i^2),
      con x_i = (q_i/q0)^(-α), y_i = S_i, w_i=1.
    - Buscamos α ∈ [α_min, α_max] por barrido 1D y escogemos el que minimiza el error LS.
    Devuelve ζ, S0 y S_fit(q).
    """
    # filtrar ventana en log10(q)
    m = (np.log10(q_um_inv) >= fit_window_log10[0]) & (np.log10(q_um_inv) <= fit_window_log10[1])
    qf = q_um_inv[m]; Sf = S[m]
    if len(qf) < 5:
        return np.nan, np.nan, np.zeros_like(S)

    # rango razonable para α (1+2ζ). ζ suele ~0..1 → α ~ 1..3
    alpha_grid = np.linspace(0.2, 4.0, 2001)  # denso
    best_err = np.inf
    best_alpha = np.nan
    best_S0 = np.nan

    # precompute logs para x_i rápido
    ln_q_ratio = np.log(qf/q0)
    for alpha in alpha_grid:
        x = np.exp(-alpha * ln_q_ratio)  # (q/q0)^(-α)
        # LS de S0: minimize ||Sf - S0*x||^2 ⇒ S0 = (x·Sf)/(x·x)
        denom = np.dot(x, x)
        if denom <= 0:
            continue
        S0 = np.dot(x, Sf) / denom
        err = np.mean((Sf - S0*x)**2)
        if err < best_err:
            best_err = err
            best_alpha = alpha
            best_S0 = S0

    if not np.isfinite(best_alpha):
        return np.nan, np.nan, np.zeros_like(S)

    zeta = 0.5*(best_alpha - 1.0)
    S_fit = best_S0 * (q_um_inv/q0)**(-best_alpha)
    return zeta, best_S0, S_fit

# ================== Fits para B(r) (en ln, más estable) ==================
def fit_B_powerlaw_ln(r_um, B_um, ln_window):
    """
    ln B = ln B0 + 2ζ ln(r/r0), con r0=1 µm => ln B = (2ζ) ln r + (ln B0).
    """
    ln_r = np.log(r_um); ln_B = np.log(B_um)
    m = (ln_r >= ln_window[0]) & (ln_r <= ln_window[1])
    if np.count_nonzero(m) < 3:
        return np.nan, np.nan, np.zeros_like(B_um)
    a, b = np.polyfit(ln_r[m], ln_B[m], 1)  # ln B = a ln r + b
    zeta = 0.5*a
    B0   = np.exp(b)  # con r0=1 µm
    B_fit = np.exp(b + a*ln_r)
    return zeta, B0, B_fit

# ================== MAIN ==================
# 1) cargar imagen y contorno
path = buscar_imagen(BASE_DIR, BASENAME, IDX, EXTS)
assert path, "No se encontró la imagen."
im = cargar_binaria(path)
H, W = im.shape

C_all = find_contours(im, level=0.5)
assert len(C_all) > 0, "No se encontró contorno."
contour = ensure_closed_no_dup(max(C_all, key=len))  # (y,x)
Y, X = contour[:,0], contour[:,1]

# 2) Método Pablo = círculo de radio medio (independiente de K)
yP, xP, uP_px, sP, PP, infoP = pablo_circle_mean_radius(Y, X, Msamples)

# --- B(r) de Pablo (denso) ---
r_px_P, B_px_P = B_of_r_dense(uP_px, PP, r_max_frac=R_MAX_FRAC)
r_um_P = r_px_P * PX_TO_UM
B_um_P = B_px_P * (PX_TO_UM**2)
zB_P, B0_P, Bfit_P = fit_B_powerlaw_ln(r_um_P, B_um_P, to_ln_window(B_FIT_LOG10))

# --- S(q) de Pablo (DFT) + fit en dominio lineal ---
q_um_inv_P, S_P = Sq_from_DFT(uP_px, PP, px_to_um=PX_TO_UM, use_hann=USE_HANN_FOR_S)
zS_P, S0_P, Sfit_P = fit_S_powerlaw_linear(q_um_inv_P, S_P, fit_window_log10=S_FIT_LOG10, q0=1.0)

# 3) Mi método (por K): curva suave + normales FFT + ray casting
res = []
for K in K_LIST:
    Ceq    = resample_by_arclength(contour, Msamples)   # (y,x)
    smooth = fft_lowpass_closed_equal_samples(Ceq, K)   # (y,x)
    yS, xS = smooth[:,0], smooth[:,1]

    # u por normales FFT
    uF_px, qyF, qxF = u_by_fft_normals_and_rays(yS, xS, Y, X)

    # eje s y perímetro propios
    ds = np.linalg.norm(np.diff(smooth, axis=0, append=smooth[:1]), axis=1)
    sK = np.concatenate(([0.0], np.cumsum(ds)))[:-1]
    PK = np.sum(ds)

    # --- B(r) denso ---
    r_px_F, B_px_F = B_of_r_dense(uF_px, PK, r_max_frac=R_MAX_FRAC)
    r_um_F = r_px_F * PX_TO_UM
    B_um_F = B_px_F * (PX_TO_UM**2)
    zB_F, B0_F, Bfit_F = fit_B_powerlaw_ln(r_um_F, B_um_F, to_ln_window(B_FIT_LOG10))

    # --- S(q) (DFT) + fit en dominio lineal ---
    q_um_inv_F, S_F = Sq_from_DFT(uF_px, PK, px_to_um=PX_TO_UM, use_hann=USE_HANN_FOR_S)
    zS_F, S0_F, Sfit_F = fit_S_powerlaw_linear(q_um_inv_F, S_F, fit_window_log10=S_FIT_LOG10, q0=1.0)

    res.append(dict(
        K=K, s=sK, P=PK, yS=yS, xS=xS,
        u_px=uF_px,
        r_um=r_um_F, B_um=B_um_F, Bfit=Bfit_F, zB=zB_F, B0=B0_F,
        q_um_inv=q_um_inv_F, S=S_F, Sfit=Sfit_F, zS=zS_F, S0=S0_F
    ))

#%%

# ================== PLOTS ==================
# 1) u(s): cada curva con SU eje s (perímetro propio)
plt.figure(figsize=(12,4))
for R in res:
    plt.plot(R["s"], R["u_px"], lw=1.1, label=f"Mi método K={R['K']}")
plt.plot(sP, uP_px, 'k--', lw=1.2, label="Pablo (círculo r0)")
plt.axhline(0, color='k', ls=':', lw=0.8)
plt.xlabel('s [px] (cada curva usa su propio perímetro)')
plt.ylabel('u(s) [px]')
plt.title('u(s): normales FFT vs. Pablo (círculo medio)')
plt.legend(ncol=2)
plt.tight_layout(); plt.show()

# 2) ln B vs ln r (µm) — ajuste en ln (robusto)
plt.figure(figsize=(12,4))
# Pablo
ln_r_P = np.log(r_um_P); ln_B_P = np.log(B_um_P)
plt.plot(ln_r_P, ln_B_P, 'k.', lw=1, label=f"Pablo (ζ={zB_P:.2f})")
plt.plot(ln_r_P, np.log(Bfit_P), 'k:', lw=0.9)
# Mi método
for R in res:
    ln_r = np.log(R["r_um"]); ln_B = np.log(R["B_um"])
    plt.plot(ln_r, ln_B, lw=1.2, label=f"Mi método K={R['K']} (ζ={R['zB']:.2f})")
    plt.plot(ln_r, np.log(R["Bfit"]), 'k:', lw=0.9)
a_ln, b_ln = to_ln_window(B_FIT_LOG10)
plt.ylim(-2,2.2)
plt.xlabel('ln r [ln µm]'); plt.ylabel('ln B(r) [ln µm²]')
plt.title(f'ln B vs ln r — fit en ln r ∈ [{a_ln:.2f}, {b_ln:.2f}]')
plt.legend(ncol=2); plt.tight_layout(); plt.show()

# 3) S(q) en escala lineal con ajuste directo del modelo (sin logs)
plt.figure(figsize=(12,4))
# Pablo
plt.plot(q_um_inv_P, S_P, 'k--', lw=1.3, label=f"Pablo datos (ζ={zS_P:.2f})")
plt.plot(q_um_inv_P, Sfit_P, 'k:', lw=0.9, label="Pablo fit")
# Mi método
for R in res:
    plt.plot(R["q_um_inv"], R["S"], lw=1.1, label=f"Mi método K={R['K']} (ζ={R['zS']:.2f})")
    plt.plot(R["q_um_inv"], R["Sfit"], 'k:', lw=0.9)
plt.xlabel('q [1/µm]'); plt.ylabel('S(q) [a.u.]')
plt.title('S(q) y ajuste S(q) ≈ S0 (q/q0)^-(1+2ζ) (fit en dominio lineal)')
plt.legend(ncol=2); plt.tight_layout(); plt.show()

# (Opcional) 4) S(q) en log–log sólo para inspección visual (el fit sigue siendo lineal)
plt.figure(figsize=(12,4))
def _maybe_crop(q, S):
    if QT_HIGH_FRAC is None: return q, S
    i0 = int(np.ceil((1.0 - QT_HIGH_FRAC) * (len(q)-1)))
    return q[i0:], S[i0:]
qP_plot, SP_plot = _maybe_crop(q_um_inv_P, S_P)
SfitP_plot = Sfit_P[-len(SP_plot):] if len(Sfit_P)>=len(SP_plot) else Sfit_P
plt.plot(qP_plot, SP_plot, 'k--', lw=1.2, label=f"Pablo (ζ={zS_P:.2f})")
plt.plot(qP_plot, SfitP_plot, 'k:', lw=0.9)
for R in res:
    qpl, Spl = _maybe_crop(R["q_um_inv"], R["S"])
    Sfitl = R["Sfit"][-len(Spl):] if len(R["Sfit"])>=len(Spl) else R["Sfit"]
    plt.plot(qpl, Spl, lw=1.0, label=f"Mi método K={R['K']}")
    plt.plot(qpl, Sfitl, 'k:', lw=0.8)
plt.xscale('log'); plt.yscale('log')
plt.xlabel('q [1/µm]'); plt.ylabel('S(q) [a.u.]')
plt.title('S(q) (log–log) — fit calculado en dominio lineal')
plt.legend(ncol=2); plt.tight_layout(); plt.show()

# 5) Visual: comparación geométrica (Pablo radial vs. normales FFT) para un K
K_show = K_LIST[0]
R0 = [R for R in res if R["K"]==K_show][0]

yc, xc, r0, thetas, r_theta = (infoP[0], infoP[1], infoP[2], infoP[3], infoP[4])
yP_ref = yc + r0*np.sin(thetas)
xP_ref = xc + r0*np.cos(thetas)
yP_dst = yc + (r0 + uP_px)*np.sin(thetas)
xP_dst = xc + (r0 + uP_px)*np.cos(thetas)

Nrm = normals_fft_from_curve(R0["yS"], R0["xS"])
pints = np.column_stack([R0["yS"], R0["xS"]]) + R0["u_px"][:,None]*Nrm

fig = plt.figure(figsize=(8,8), facecolor='white'); ax = fig.add_subplot(111)
ax.plot(X, Y, color='#2E86C1', lw=1.6, label='Contorno real')
ax.plot(xP_ref, yP_ref, color='#8E44AD', lw=1.6, label='Pablo: círculo r0')
stepP = max(1, Msamples//360)
for i in range(0, Msamples, stepP):
    ax.plot([xP_ref[i], xP_dst[i]], [yP_ref[i], yP_dst[i]], color='#8E44AD', lw=0.9, alpha=0.7)
ax.plot(R0["xS"], R0["yS"], color='#16A085', lw=1.6, label=f'Suave (K={K_show})')
stepF = max(1, len(R0["xS"])//1200)
for i in range(0, len(R0["xS"]), stepF):
    ax.plot([R0["xS"][i], pints[i,1]], [R0["yS"][i], pints[i,0]], color='red', lw=0.9)
ax.set_aspect('equal', adjustable='box'); ax.invert_yaxis()
ax.set_title('Pablo (radial, violeta) vs. normales FFT (rojo)')
ax.legend(loc='lower right', frameon=False)
plt.tight_layout(); plt.show()

