# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from skimage import measure, filters, morphology, util, color, io
from skimage.draw import polygon
from scipy.ndimage import distance_transform_edt
import os, glob

# ================== util geom + fourier ==================
def _ensure_closed(poly):
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly

def _polygon_area(poly):
    P = _ensure_closed(poly)
    x, y = P[:,0], P[:,1]
    return 0.5*np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])

def _perimeter(poly):
    d = np.linalg.norm(np.diff(_ensure_closed(poly), axis=0), axis=1)
    return d.sum()

def _resample_closed(poly, M):
    P = _ensure_closed(poly)
    segs = np.linalg.norm(np.diff(P, axis=0), axis=1)
    Lcum = np.concatenate([[0], np.cumsum(segs)])
    L = Lcum[-1]
    s_new = np.linspace(0, L, M+1)[:-1]
    idx = np.searchsorted(Lcum, s_new, side='right') - 1
    idx = np.clip(idx, 0, len(segs)-1)
    t = (s_new - Lcum[idx]) / (segs[idx] + 1e-12)
    Q = P[idx] + (P[idx+1] - P[idx]) * t[:, None]
    return Q, L

def _fft_lowpass_closed(curve, K_keep):
    z = curve[:,0] + 1j*curve[:,1]
    Z = np.fft.fft(z); M = len(z)
    keep = np.zeros(M, dtype=bool); keep[0] = True
    for k in range(1, K_keep+1):
        keep[k%M] = True; keep[-k%M] = True
    z_s = np.fft.ifft(np.where(keep, Z, 0))
    return np.column_stack([z_s.real, z_s.imag])

def rasterize_closed(curve, shape):
    C = _ensure_closed(curve)[:-1]
    rr, cc = polygon(C[:,1], C[:,0], shape=shape)  # (row=y, col=x)
    m = np.zeros(shape, dtype=bool); m[rr, cc] = True
    return m

# ============== contorno real desde imagen ==============
def extract_largest_contour(mask):
    if mask.dtype != bool:
        mask = mask > 0
    mask2 = morphology.remove_small_holes(mask, area_threshold=64)
    mask2 = morphology.remove_small_objects(mask2, min_size=64)
    conts = measure.find_contours(mask2.astype(float), 0.5)
    if not conts:
        raise ValueError("No se encontró contorno.")
    conts_xy = [c[:, ::-1] for c in conts]  # (x,y)
    conts_xy = [c if _polygon_area(c) > 0 else np.flipud(c) for c in conts_xy]
    perims = [ _perimeter(c) for c in conts_xy ]
    return conts_xy[int(np.argmax(perims))], mask2

# ============== fajas (máscara ref vs real) ==============
def bands_from_C1_to_C2(C1b, C2b, thr=0):
    C1b = (C1b.astype(float) > thr); C2b = (C2b.astype(float) > thr)
    growth = C2b & (~C1b)
    shrink = C1b & (~C2b)
    return dict(C1b=C1b, C2b=C2b, growth=growth, shrink=shrink)

# ======= gradiente (normal) vía distancia firmada =======
def signed_distance_and_grad(M_ref):
    """
    Construye distancia firmada d (positivo afuera, negativo adentro) y su gradiente continuo.
    La dirección de la normal en cada punto será n_hat = grad(d)/||grad(d)||.
    """
    d_in  = distance_transform_edt(M_ref)
    d_out = distance_transform_edt(~M_ref)
    d = d_out.astype(np.float32)
    d[M_ref] = -d_in[M_ref].astype(np.float32)

    # gradiente en convención (gy, gx) = deriv w.r.t (row=y, col=x)
    gy, gx = np.gradient(d)
    return d, gx, gy

def bilinear_sample(arr, xy):
    """
    Bilinear en coords (x,y) flotantes. arr indexado [row=y, col=x].
    """
    x, y = xy
    H, W = arr.shape[:2]
    x0 = int(np.floor(x)); x1 = x0 + 1
    y0 = int(np.floor(y)); y1 = y0 + 1
    x0c = np.clip(x0, 0, W-1); x1c = np.clip(x1, 0, W-1)
    y0c = np.clip(y0, 0, H-1); y1c = np.clip(y1, 0, H-1)
    wx = x - x0; wy = y - y0
    v00 = arr[y0c, x0c]; v10 = arr[y0c, x1c]
    v01 = arr[y1c, x0c]; v11 = arr[y1c, x1c]
    return (1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + (1-wx)*wy*v01 + wx*wy*v11

# ========== intersección rayo-normal con polígono ==========
def _ray_poly_intersection_one_side(p, n_hat, poly, tmin=1e-6):
    """
    Interseca p + t n (t>tmin) con cada segmento de poly (cerrado).
    Devuelve (t,q) del cruce más cercano o (np.nan, p) si no hay.
    """
    P = _ensure_closed(poly)
    A = P[:-1]; B = P[1:]
    best_t = np.inf; best_q = p
    for a, b in zip(A, B):
        r = b - a
        M = np.array([[n_hat[0], -r[0]], [n_hat[1], -r[1]]], dtype=float)
        rhs = a - p
        det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
        if abs(det) < 1e-12:
            continue
        invM = np.array([[ M[1,1], -M[0,1]], [-M[1,0], M[0,0]]], float)/det
        t, s = invM @ rhs
        if t > tmin and 0.0 <= s <= 1.0:
            if t < best_t:
                best_t = t
                best_q = p + t*n_hat
    if not np.isfinite(best_t):
        return np.nan, p
    return float(best_t), best_q

def _distance_continua_por_normal(C_ref, C_real_res, N_dir):
    """
    Para cada punto p de C_ref, dispara rayo en +N_dir[j] y en -N_dir[j],
    elige el cruce más cercano en |t| y asigna signo según el sentido elegido.
    """
    M = len(C_ref)
    u = np.zeros(M, float)
    Q = np.zeros_like(C_ref)
    for j in range(M):
        p = C_ref[j]; n = N_dir[j]

        t_pos, q_pos = _ray_poly_intersection_one_side(p, n,  C_real_res, tmin=1e-6)
        t_neg, q_neg = _ray_poly_intersection_one_side(p, -n, C_real_res, tmin=1e-6)

        cand = []
        if np.isfinite(t_pos): cand.append((+t_pos, q_pos, +1))
        if np.isfinite(t_neg): cand.append((-t_neg, q_neg, -1))  # nota: distancia negativa

        if cand:
            cand.sort(key=lambda c: abs(c[0]))
            u[j], Q[j], _ = cand[0]
        else:
            # fallback raro: proyección al segmento más cercano y firmo por n·(q-p)
            A = C_real_res[:-1]; B = C_real_res[1:]
            AB = B - A; AP = p - A
            ab2 = np.einsum('ij,ij->i', AB, AB) + 1e-12
            tseg = np.clip(np.einsum('ij,ij->i', AP, AB)/ab2, 0.0, 1.0)
            Qs = A + AB*tseg[:,None]
            d2 = np.einsum('ij,ij->i', Qs - p, Qs - p)
            i = int(np.argmin(d2))
            q = Qs[i]
            sgn = np.sign(np.dot(q - p, n))
            u[j] = sgn * np.linalg.norm(q - p)
            Q[j] = q
    return u, Q

# ================== B(r) (opcional) ==================
def compute_B_of_r(u, L):
    u = np.asarray(u, float)
    M = len(u); ds = L / M
    max_m = M // 2
    r = np.arange(1, max_m+1) * ds
    B = np.empty_like(r, dtype=float)
    for idx, m in enumerate(range(1, max_m+1)):
        diff = np.roll(u, -m) - u
        B[idx] = np.mean(diff*diff)
    return r, B

def fit_loglog(x, y, log10_range=None):
    x = np.asarray(x); y = np.asarray(y)
    m = (x>0) & (y>0)
    lx, ly = np.log10(x[m]), np.log10(y[m])
    if log10_range is not None:
        lo, hi = log10_range
        sel = (lx >= lo) & (lx <= hi)
        lx, ly = lx[sel], ly[sel]
    if len(lx) < 2: return np.nan, np.nan
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    return slope, intercept

# ================== Runner principal ==================
def run_B_via_fourier_NORMALFIELD_on_image(
    im,
    Msamples=4096,
    K_keep=30,
    b_fit_log10=(0.0, 0.8),
    do_plots=True,
    draw_every=1   # 1 = dibuja todas las fajas
):
    # 0) máscara
    img = im.copy()
    if img.dtype == bool:
        mask = img; thr_used = None
    else:
        img = util.img_as_float(img)
        if img.ndim == 3: img = color.rgb2gray(img)
        thr_used = filters.threshold_otsu(img)
        mask = img >= thr_used

    # 1) contorno real y resampleo
    C_real, _ = extract_largest_contour(mask)
    C_real_res, _ = _resample_closed(C_real, Msamples)

    # 2) curva suave por Fourier
    C_ref = _fft_lowpass_closed(C_real_res, K_keep=K_keep)
    L_ref = _perimeter(C_ref)
    z_axis = np.linspace(0, L_ref, Msamples, endpoint=False)

    # 3) fajas → dirección de normal (campo normal desde gradiente de d_signed)
    H, W = mask.shape
    M_ref  = rasterize_closed(C_ref,      shape=(H, W))
    M_real = rasterize_closed(C_real_res, shape=(H, W))
    bands = bands_from_C1_to_C2(M_ref, M_real)  # puede servir si querés checks growth/shrink

    d_signed, gx, gy = signed_distance_and_grad(M_ref)

    # campo normal continuo: n_hat = grad(d)/||grad(d)||
    N_dir = np.zeros_like(C_ref)
    for j, p in enumerate(C_ref):
        # muestreo bilinear de gradiente en (x,y)
        g = np.array([bilinear_sample(gx, (p[0], p[1])),
                      bilinear_sample(gy, (p[0], p[1]))], float)
        nv = np.linalg.norm(g)
        if nv < 1e-8:
            # fallback: normal geométrica local usando vecinos
            # (robusto para casos puntuales donde grad ~ 0)
            i0 = (j-1) % Msamples; i1 = (j+1) % Msamples
            t = C_ref[i1] - C_ref[i0]
            t /= np.linalg.norm(t) + 1e-12
            n = np.array([t[1], -t[0]])
            if _polygon_area(C_ref) < 0: n = -n
            N_dir[j] = n
        else:
            N_dir[j] = g / nv  # apunta “hacia afuera” de C_ref por construcción

    # 4) distancia continua por intersección rayo-normal (elige ± según más cercano)
    u, Q = _distance_continua_por_normal(C_ref, C_real_res, N_dir)

    # ================== plots solicitados ==================
    if do_plots:
        # (A) TODAS las fajas coloreadas por u
        fig, ax = plt.subplots(1,1, figsize=(6.4,6.4))
        ax.plot(C_real_res[:,0], C_real_res[:,1], 'k-', lw=1, alpha=0.6, label='contorno real')
        ax.plot(C_ref[:,0],      C_ref[:,1],      '-',  lw=2, color='tab:blue', label='curva suave')
        # colorbar centrado en 0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=np.min(u), vmax=np.max(u))
        cmap = plt.get_cmap('coolwarm')
        for j in range(0, Msamples, draw_every):
            col = cmap(norm(u[j]))
            ax.plot([C_ref[j,0], Q[j,0]], [C_ref[j,1], Q[j,1]], '-', lw=1.0, color=col)
        ax.set_aspect('equal'); ax.set_title('Fajas (todas) coloreadas por u')
        ax.legend(loc='lower right', fontsize=8)
        # crear colorbar manual con scatter invisible
        sc = ax.scatter([np.nan],[np.nan], c=[0], cmap=cmap, norm=norm)
        cbar = plt.colorbar(sc, ax=ax); cbar.set_label('u [px]')
        plt.tight_layout(); plt.show()

        # (B) Pares de puntos: contorno suave y puntos de cruce
        fig2, ax2 = plt.subplots(1,1, figsize=(6.4,6.4))
        ax2.plot(C_ref[:,0],      C_ref[:,1],      '-', lw=1.5, label='curva suave')
        ax2.plot(C_real_res[:,0], C_real_res[:,1], '-', lw=1.0, alpha=0.6, label='contorno real')
        ax2.scatter(Q[:,0], Q[:,1], s=8, c=u, cmap='coolwarm', norm=norm, label='puntos de cruce')
        ax2.set_aspect('equal'); ax2.set_title('Pares (p en suave) → (q en real)')
        ax2.legend(loc='lower right', fontsize=8)
        cbar = plt.colorbar(ax2.collections[-1], ax=ax2); cbar.set_label('u [px]')
        plt.tight_layout(); plt.show()

        # (C) (opcional) u(z) y B(r)
        fig3, ax3 = plt.subplots(1,2, figsize=(11,4))
        ax3[0].plot(z_axis, u, lw=1); ax3[0].axhline(0, color='k', lw=0.8, alpha=0.5)
        ax3[0].set_xlabel('z [px de perímetro]'); ax3[0].set_ylabel('u(z) [px]')
        ax3[0].set_title('u(z) continuo (intersección por rayos normales)')
        r, B = compute_B_of_r(u, L_ref)
        ax3[1].loglog(r, B, '.', ms=4)
        ax3[1].set_xlabel('r [px]'); ax3[1].set_ylabel('B(r)')
        ax3[1].set_title('B(r)')
        plt.tight_layout(); plt.show()

    # slope opcional
    r, B = compute_B_of_r(u, L_ref)
    slope, intercept = fit_loglog(r, B, b_fit_log10)
    zeta = slope/2 if np.isfinite(slope) else np.nan

    return dict(
        C_real=C_real_res, C_ref=C_ref,
        N_dir=N_dir, Q=Q, u=u, z=np.linspace(0, L_ref, Msamples, endpoint=False),
        r=r, B=B, slope=slope, zeta=zeta,
        L=L_ref, Msamples=Msamples, K_keep=K_keep
    )

# ================== carga de imagen (tu helper) ==================
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

# ================== ejemplo de uso ==================
if __name__ == "__main__":
    BASE_DIR = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe\080"
    BASENAME = "Bin-P8137-150Oe-3ms-"
    IDX      = 5
    EXTS     = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]

    path = buscar_imagen(BASE_DIR, BASENAME, IDX, EXTS)
    if not path:
        raise FileNotFoundError("No encontré la imagen. Revisá BASE_DIR/BASENAME/EXTS/IDX.")
    print("Usando:", path)
    im = cargar_binaria(path)

    out = run_B_via_fourier_NORMALFIELD_on_image(
        im,
        Msamples=1024,
        K_keep=2,
        b_fit_log10=(0.0, 0.8),
        do_plots=True,
        draw_every=1     # 1 = dibuja TODAS las fajas
    )
    print("Perímetro L =", out["L"], "px")
    print("ζ (aprox)   =", out["zeta"])

#%%

# --- Comparación de B(r) y zeta para distintos K_keep ---
import numpy as np
import matplotlib.pyplot as plt

# A) Parámetros de barrido
K_LIST    = [1,2,3,4,5,6,7,8,9,10]   # valores de K_keep a comparar
MSAMPLES  = 1024                        # muestreo en arco (bajá si querés más velocidad)
FIT_RANGE = (0.0, 0.8)                   # décadas para el fit de B(r) (log10 r en [10^a,10^b])

# B) Helpers: usamos funciones ya definidas arriba
req_funcs = [
    '_resample_closed', '_fft_lowpass_closed', '_perimeter', 'rasterize_closed',
    'bands_from_C1_to_C2', 'signed_distance_and_grad', 'bilinear_sample',
    '_distance_continua_por_normal', 'compute_B_of_r', 'fit_loglog',
    'extract_largest_contour'
]
for f in req_funcs:
    assert f in globals(), f"Falta definir la función: {f} (corré la celda anterior)."

# C) Preparación: contorno real resampleado UNA sola vez
#    - Usamos 'out' si ya lo tenés; si no, extraemos del 'im' binario.
if 'out' in globals() and isinstance(out, dict) and ('C_real' in out):
    C_real_base = out['C_real']         # ya resampleado, pero lo re-resampleamos a MSAMPLES por consistencia
    # Recupero máscara shape desde la imagen 'im'
    assert 'im' in globals(), "No encuentro 'im'. Cargá la imagen como antes."
    mask_shape = im.shape[:2]
else:
    # Extraer contorno real desde 'im'
    assert 'im' in globals(), "No encuentro 'im'. Cargá la imagen como antes."
    if im.dtype == bool:
        mask = im
    else:
        from skimage import util, color, filters
        img = util.img_as_float(im)
        if img.ndim == 3:
            img = color.rgb2gray(img)
        thr = filters.threshold_otsu(img)
        mask = img >= thr
    C_real_base, _ = extract_largest_contour(mask)
    mask_shape = mask.shape

# Resampleo del contorno real al MSAMPLES elegido
C_real_res, _ = _resample_closed(C_real_base, MSAMPLES)

def compute_B_for_K(C_real_res, mask_shape, K_keep, fit_range):
    """
    Pipeline rápido para un K dado:
      - suaviza C_real_res -> C_ref(K)
      - obtiene normales desde gradiente de distancia firmada a C_ref
      - interseca rayos normales con contorno real (geométrico, continuo)
      - calcula B(r) y fit para zeta
    """
    # curva suave
    C_ref = _fft_lowpass_closed(C_real_res, K_keep=K_keep)
    L_ref = _perimeter(C_ref)

    # máscaras para fajas (solo para construir distancia firmada a C_ref)
    H, W = mask_shape
    M_ref  = rasterize_closed(C_ref,      shape=(H, W))
    M_real = rasterize_closed(C_real_res, shape=(H, W))

    # distancia firmada y gradiente -> campo normal punto a punto
    d_signed, gx, gy = signed_distance_and_grad(M_ref)
    N_dir = np.zeros_like(C_ref)
    for j, p in enumerate(C_ref):
        g = np.array([bilinear_sample(gx, (p[0], p[1])),
                      bilinear_sample(gy, (p[0], p[1]))], float)
        nv = np.linalg.norm(g)
        if nv < 1e-8:
            # fallback: normal geométrica local
            i0 = (j-1) % len(C_ref); i1 = (j+1) % len(C_ref)
            t = C_ref[i1] - C_ref[i0]; t /= (np.linalg.norm(t) + 1e-12)
            n = np.array([t[1], -t[0]])
            # asegurá coherencia exterior si querés, pero para el rayo elegimos ± automáticamente
            N_dir[j] = n
        else:
            N_dir[j] = g / nv

    # distancia continua por rayo normal (elige ± el cruce más cercano)
    u, Q = _distance_continua_por_normal(C_ref, C_real_res, N_dir)

    # B(r) y fit
    r, B = compute_B_of_r(u, L_ref)
    slope, intercept = fit_loglog(r, B, log10_range=fit_range)
    zeta = slope/2 if np.isfinite(slope) else np.nan

    return dict(K=K_keep, C_ref=C_ref, u=u, r=r, B=B,
                slope=slope, zeta=zeta, L=L_ref)

# D) Barrido en K
results = []
for K in K_LIST:
    resK = compute_B_for_K(C_real_res, mask_shape, K_keep=K, fit_range=FIT_RANGE)
    results.append(resK)
    print(f"K={K:>4}  slope={resK['slope']:.3f}  zeta≈{resK['zeta']:.3f}")

# E) Plots comparativos
# E1) B(r) superpuestas (log–log)
plt.figure(figsize=(7.5,5.2))
for res in results:
    r, B, K = res['r'], res['B'], res['K']
    plt.loglog(r, B, '.', ms=3, alpha=0.8, label=f"K={K}  ζ≈{res['zeta']:.3f}")
plt.xlabel('r [px de perímetro]'); plt.ylabel('B(r)')
plt.title('Comparación B(r) para distintos K (curva suave)')
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# E2) ζ vs K
plt.figure(figsize=(6.4,4.2))
Ks = [res['K'] for res in results]
Zs = [res['zeta'] for res in results]
plt.plot(Ks, Zs, 'o-', lw=1.5)
plt.xlabel('K (modos conservados en la curva suave)')
plt.ylabel('ζ (≈ slope/2)')
plt.title('Exponente ζ vs K')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%

# === COMPARACIÓN u(z): Tu método (normales+rayos) con varios K vs Pablo (ρ·θ), MISMO ANCLA ===
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from skimage import io, color, measure
from skimage.draw import polygon
from scipy.ndimage import distance_transform_edt

# -------------------- Parámetros --------------------
K_LIST    = [1,3,5,10,20]   # K (±K) para la curva suave (tu método)
N_S       = 1024                        # puntos por arclonga para tu método (subí/bajá por velocidad)
N_TH      = 1024                        # puntos angulares para ρ·θ
FIT_RANGE = (0.1,0.8)                 # décadas en log10(r) para ajuste (en px)
MSHAPE_FROM_IM = True                   # si True, usa im.shape para rasterizado; si False, usa bbox del contorno

# -------------------- Helpers mínimos (auto-contenidos) --------------------
def cargar_binaria(path):
    im = io.imread(path)
    if im.ndim == 3:
        im = color.rgb2gray(im)
        return (im > 0.5).astype(bool)
    if np.issubdtype(im.dtype, np.integer):
        return (im > (np.iinfo(im.dtype).max/2)).astype(bool)
    return (im > 0.5).astype(bool)

def contorno_principal(binaria, level=0.5):
    conts = measure.find_contours(binaria.astype(float), level=level)
    if not conts: return None
    C = max(conts, key=lambda c: c.shape[0])  # (y,x)
    if np.linalg.norm(C[0] - C[-1]) > 1.0:
        C = np.vstack([C, C[0]])
    y, x = C[:,0], C[:,1]
    return x, y

def polygon_centroid(x, y):
    if np.hypot(x[0]-x[-1], y[0]-y[-1]) < 1e-12: x, y = x[:-1], y[:-1]
    X, Y = np.r_[x, x[0]], np.r_[y, y[0]]
    cross = X[:-1]*Y[1:] - X[1:]*Y[:-1]
    A = 0.5*np.sum(cross)
    if abs(A) < 1e-12: return float(np.mean(x)), float(np.mean(y))
    Cx = (1/(6*A))*np.sum((X[:-1]+X[1:])*cross)
    Cy = (1/(6*A))*np.sum((Y[:-1]+Y[1:])*cross)
    return float(Cx), float(Cy)

def resample_xy_arclength_from_anchor(x, y, N=2048, i_anchor=0):
    if np.hypot(x[0]-x[-1], y[0]-y[-1]) < 1e-12: x, y = x[:-1], y[:-1]
    x = np.roll(x, -i_anchor); y = np.roll(y, -i_anchor)
    seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    s   = np.r_[0.0, np.cumsum(seg[:-1])]
    L   = s[-1] + seg[-1]
    t   = s / L
    t_ext = np.r_[t, 1.0]; x_ext = np.r_[x, x[0]]; y_ext = np.r_[y, y[0]]
    tgrid = np.linspace(0, 1, N, endpoint=False)
    xs = np.interp(tgrid, t_ext, x_ext); ys = np.interp(tgrid, t_ext, y_ext)
    ds = L / N
    return xs, ys, ds, L

def fft_lowpass_complex(z, K):
    if K <= 0: return z
    Z = np.fft.fft(z); N = Z.size
    Zf = np.zeros_like(Z)
    kmax = min(K, N//2)
    Zf[:kmax+1] = Z[:kmax+1]
    if kmax > 0: Zf[-kmax:] = Z[-kmax:]
    return np.fft.ifft(Zf)

def rasterize_closed_xy(x, y, shape):
    # (x,y) -> máscara booleana
    rr, cc = polygon(y, x, shape=shape)
    m = np.zeros(shape, dtype=bool); m[rr, cc] = True
    return m

def signed_distance_and_grad(mask):
    d_in  = distance_transform_edt(mask)
    d_out = distance_transform_edt(~mask)
    d = d_out.astype(np.float32); d[mask] = -d_in[mask].astype(np.float32)
    gy, gx = np.gradient(d)        # grad(d) = (gx, gy) en coords (x,y)
    return d, gx, gy

def bilinear_sample(arr, x, y):
    H, W = arr.shape[:2]
    x0 = int(np.floor(x)); x1 = x0 + 1
    y0 = int(np.floor(y)); y1 = y0 + 1
    x0c = np.clip(x0, 0, W-1); x1c = np.clip(x1, 0, W-1)
    y0c = np.clip(y0, 0, H-1); y1c = np.clip(y1, 0, H-1)
    wx = x - x0; wy = y - y0
    v00 = arr[y0c, x0c]; v10 = arr[y0c, x1c]
    v01 = arr[y1c, x0c]; v11 = arr[y1c, x1c]
    return (1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + (1-wx)*wy*v01 + wx*wy*v11

def ray_poly_intersection_one_side(p, n_hat, xR, yR, tmin=1e-6):
    # segmentos del contorno real
    X = np.r_[xR, xR[0]]; Y = np.r_[yR, yR[0]]
    best_t = np.inf; best_q = p.copy()
    for i in range(len(X)-1):
        a = np.array([X[i], Y[i]]); b = np.array([X[i+1], Y[i+1]])
        r = b - a
        M = np.array([[n_hat[0], -r[0]], [n_hat[1], -r[1]]], float)
        rhs = a - p
        det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
        if abs(det) < 1e-12: continue
        invM = np.array([[ M[1,1], -M[0,1]], [-M[1,0], M[0,0]]], float)/det
        t, s = invM @ rhs
        if t > tmin and 0.0 <= s <= 1.0:
            if t < best_t:
                best_t = t; best_q = p + t*n_hat
    if not np.isfinite(best_t): return np.nan, p
    return float(best_t), best_q

def distance_continua_por_normal(xRef, yRef, xReal, yReal, N_dir):
    M = len(xRef)
    u = np.zeros(M, float); qx = np.zeros(M, float); qy = np.zeros(M, float)
    for j in range(M):
        p = np.array([xRef[j], yRef[j]]); n = N_dir[j]
        tpos, qpos = ray_poly_intersection_one_side(p, n,  xReal, yReal)
        tneg, qneg = ray_poly_intersection_one_side(p, -n, xReal, yReal)
        cand = []
        if np.isfinite(tpos): cand.append((+tpos, qpos, +1))
        if np.isfinite(tneg): cand.append((-tneg, qneg, -1))
        if cand:
            cand.sort(key=lambda c: abs(c[0]))
            u[j], q = cand[0][0], cand[0][1]
            qx[j], qy[j] = q[0], q[1]
        else:
            u[j]  = 0.0; qx[j], qy[j] = p[0], p[1]
    return u, qx, qy

def compute_B_wrap(u):
    u = np.asarray(u, float); N = u.size
    B = np.empty(N-1, float); idx = np.arange(N)
    for k in range(1, N):
        diff = u[(idx+k)%N] - u
        B[k-1] = np.mean(diff*diff)
    return B

def linfit_loglog(x, y):
    m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
    if np.count_nonzero(m) < 3: return np.nan, np.nan, np.nan
    lx, ly = np.log10(x[m]), np.log10(y[m])
    p = np.polyfit(lx, ly, 1)
    yhat = np.polyval(p, lx)
    ss_res = np.sum((ly - yhat)**2); ss_tot = np.sum((ly - ly.mean())**2)
    R2 = 1.0 - ss_res/ss_tot if ss_tot>0 else np.nan
    return float(p[0]), float(p[1]), R2

def raycast_r_of_theta(x, y, xc, yc, Ntheta=2048, theta0=0.0, eps=1e-12):
    X, Y = np.r_[x, x[0]], np.r_[y, y[0]]
    theta = (theta0 + np.linspace(0, 2*np.pi, Ntheta, endpoint=False)) % (2*np.pi)
    ct, st = np.cos(theta), np.sin(theta)
    r = np.empty(Ntheta, float)
    for k in range(Ntheta):
        dx, dy = ct[k], st[k]; hits = []
        for i in range(len(X)-1):
            px, py = X[i], Y[i]; sx, sy = X[i+1]-px, Y[i+1]-py
            det = dx*(-sy) - dy*(-sx)
            if abs(det) < eps: continue
            bx, by = px - xc, py - yc
            t = (bx*(-sy) - by*(-sx)) / det
            u = (dx*by - dy*bx) / det
            if (t>=0) and (0.0<=u<=1.0): hits.append(t)
        r[k] = max(hits) if hits else np.nan
    return theta, r

# -------------------- Entrada: tomamos 'im' si ya existe, si no, usar path --------------------
if 'im' in globals():
    binaria = (im > 0) if im.dtype != bool else im.astype(bool)
    H, W = binaria.shape[:2]
else:
    # EDITÁ estoooo si querés correr directo sin 'im':
    BASE_DIR = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe\080"
    BASENAME = "Bin-P8137-080Oe-100ms-"
    IDX      = 5
    EXTS     = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]
    import os, glob
    def buscar_imagen(base_dir, basename, idx, exts):
        for ext in exts:
            p = os.path.join(base_dir, f"{basename}{idx}{ext}")
            if os.path.exists(p): return p
        cand = glob.glob(os.path.join(base_dir, f"{basename}{idx}*"))
        cand = [p for p in cand if os.path.splitext(p)[1].lower() in exts]
        return cand[0] if cand else ""
    path = buscar_imagen(BASE_DIR, BASENAME, IDX, EXTS)
    assert path, "No encontré la imagen (ajustá BASE_DIR/BASENAME/IDX/EXTS)."
    binaria = cargar_binaria(path)
    H, W = binaria.shape[:2]

# -------------------- Contorno + ancla común --------------------
x_raw, y_raw = contorno_principal(binaria, level=0.5)
xc, yc = polygon_centroid(x_raw, y_raw)
i_anchor = int(np.argmax(x_raw - xc))  # “partiendo del mismo punto”
theta_anchor = float(np.mod(np.arctan2(y_raw[i_anchor]-yc, x_raw[i_anchor]-xc), 2*np.pi))

# Remuestreo del contorno REAL anclado
xs, ys, ds_raw, L_raw = resample_xy_arclength_from_anchor(x_raw, y_raw, N=N_S, i_anchor=i_anchor)
z_raw = xs + 1j*ys

# -------------------- Método de Pablo (ρ·θ) en eje z=ρ0θ (mismo ancla) --------------------
theta_th, rtheta = raycast_r_of_theta(x_raw, y_raw, xc, yc, Ntheta=N_TH, theta0=theta_anchor)
rho0 = float(np.nanmean(rtheta))
u_th = rtheta - rho0
dtheta = (2*np.pi)/N_TH
dz     = rho0 * dtheta
z_axis = np.arange(N_TH, dtype=float) * dz           # eje común 0..2πρ0

# -------------------- Tu método para varios K (normales por fajas + rayos) --------------------
if MSHAPE_FROM_IM:
    shape_mask = (H, W)
else:
    # bbox ajustada al contorno (p/rasters compactos)
    xmin, xmax = int(np.floor(x_raw.min()-3)), int(np.ceil(x_raw.max()+3))
    ymin, ymax = int(np.floor(y_raw.min()-3)), int(np.ceil(y_raw.max()+3))
    W = max(8, xmax - xmin + 1); H = max(8, ymax - ymin + 1)
    shape_mask = (H, W)
    # desplazamos coords si quisiéramos empaquetar… (lo dejo desactivado para evitar traslados)

results = []
for K in K_LIST:
    # referencia suave (±K)
    z_ref = fft_lowpass_complex(z_raw, K)
    x_ref, y_ref = np.real(z_ref), np.imag(z_ref)

    # máscaras para firmar distancia y obtener gradiente (dirección de normal por fajas)
    M_ref  = rasterize_closed_xy(x_ref, y_ref, shape=shape_mask)

    # campo normal desde gradiente de distancia firmada
    _, gx, gy = signed_distance_and_grad(M_ref)
    N_dir = np.zeros((len(x_ref), 2), float)
    for j in range(len(x_ref)):
        g = np.array([bilinear_sample(gx, x_ref[j], y_ref[j]),
                      bilinear_sample(gy, x_ref[j], y_ref[j])], float)
        nv = np.linalg.norm(g)
        if nv < 1e-8:
            # fallback: normal geométrica local
            j0 = (j-1) % len(x_ref); j1 = (j+1) % len(x_ref)
            t = np.array([x_ref[j1]-x_ref[j0], y_ref[j1]-y_ref[j0]])
            t /= (np.linalg.norm(t) + 1e-12)
            n = np.array([t[1], -t[0]])
            N_dir[j] = n
        else:
            N_dir[j] = g / nv

    # distancia continua por intersección con contorno real (geométrico)
    u_k, qx_k, qy_k = distance_continua_por_normal(x_ref, y_ref, xs, ys, N_dir)

    # mapear al eje común z=ρ0θ (MISMA ancla): z_k = ρ0*(θ_k - θ_anchor) mod 2πρ0
    theta_k = np.unwrap(np.arctan2(y_ref - yc, x_ref - xc))
    z_k     = rho0 * (theta_k - theta_anchor)   # puede salirse; traigo a [0, 2πρ0)
    P_ref   = 2*np.pi*rho0
    z_k_wrapped = np.mod(z_k, P_ref)

    # regrilla u_k a la malla uniforme z_axis
    idx = np.argsort(z_k_wrapped)
    z_sorted = z_k_wrapped[idx]; u_sorted = u_k[idx]
    z_ext = np.r_[z_sorted, z_sorted[0] + P_ref]
    u_ext = np.r_[u_sorted, u_sorted[0]]
    u_on_common = np.interp(z_axis, z_ext, u_ext)

    # B(r) en el mismo espaciamiento (dz uniforme)
    B_k = compute_B_wrap(u_on_common)
    r_k = np.arange(1, len(u_on_common), dtype=float) * dz

    # ajuste en FIT_RANGE
    m = (r_k > 0) & (r_k >= 10**FIT_RANGE[0]) & (r_k <= 10**FIT_RANGE[1]) & (B_k > 0)
    slope, icpt, R2 = linfit_loglog(r_k[m], B_k[m])
    zeta = 0.5 * slope if np.isfinite(slope) else np.nan

    results.append(dict(K=K, u=u_on_common, B=B_k, r=r_k, slope=slope, zeta=zeta, R2=R2))
    print(f"[K={K:>3}] slope={slope:.3f}  ζ≈{zeta:.3f}  R²={R2:.3f}")

# Pablo (ρ·θ): B y ajuste en mismo eje
B_th = compute_B_wrap(u_th)
r_th = np.arange(1, len(u_th), dtype=float) * dz
mth  = (r_th > 0) & (r_th >= 10**FIT_RANGE[0]) & (r_th <= 10**FIT_RANGE[1]) & (B_th > 0)
slope_th, icpt_th, R2_th = linfit_loglog(r_th[mth], B_th[mth])
zeta_th = 0.5 * slope_th if np.isfinite(slope_th) else np.nan
print(f"[Pablo  ] slope={slope_th:.3f}  ζ≈{zeta_th:.3f}  R²={R2_th:.3f}")

# -------------------- PLOTS --------------------
# (1) u(z) superpuestos (MISMO eje z y MISMA ancla)
plt.figure(figsize=(8.2,4.6))
norm = TwoSlopeNorm(vcenter=0.0, vmin=min([np.min(r['u']) for r in results]+[np.min(u_th)]),
                    vmax=max([np.max(r['u']) for r in results]+[np.max(u_th)]))
for r in results:
    plt.plot(z_axis, r['u'], lw=1.1, label=f"K={r['K']} · ζ≈{r['zeta']:.3f}")
plt.plot(z_axis, u_th, '--', lw=1.4, label=f"Pablo (ρ·θ) · ζ≈{zeta_th:.3f}")
plt.xlabel("z = ρ₀ θ  [px]"); plt.ylabel("u(z) [px]")
plt.title("u(z) — MISMO ancla y eje común")
plt.legend(ncol=2, fontsize=8); plt.tight_layout()
plt.show()

# (2) B(r) en log–log
plt.figure(figsize=(8.2,4.8))
for r in results:
    plt.loglog(r['r'], r['B'], '.', ms=3, alpha=0.85, label=f"K={r['K']} (ζ≈{r['zeta']:.3f})")
plt.loglog(r_th, B_th, '.', ms=3, alpha=0.9, label=f"Pablo ρ·θ (ζ≈{zeta_th:.3f})")
plt.xlabel("r [px]"); plt.ylabel(r"B(r) [px$^2$]")
plt.title("B(r) — eje z=ρ₀θ común")
plt.legend(ncol=2, fontsize=8); plt.tight_layout()
plt.show()

# (3) ζ vs K, con Pablo como referencia
plt.figure(figsize=(6.2,4.2))
Ks = [r['K'] for r in results]; Zs = [r['zeta'] for r in results]
plt.plot(Ks, Zs, 'o-', lw=1.6, label='Tu método (ζ vs K)')
plt.axhline(zeta_th, ls='--', lw=1.2, label=f'Pablo (ρ·θ): ζ≈{zeta_th:.3f}')
plt.xlabel('K (modos en referencia suave)'); plt.ylabel('ζ')
plt.title('Exponente ζ vs K (ancla y eje comunes)')
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.show()

#%%
# === u(z) en un único plot, desplazados; K ascendente y Pablo ARRIBA ===
import numpy as np
import matplotlib.pyplot as plt

assert 'results' in globals() and len(results) > 0, "Falta 'results'."
assert 'z_axis' in globals(), "Falta 'z_axis'."
assert 'u_th' in globals() and 'zeta_th' in globals(), "Faltan 'u_th' y/o 'zeta_th'."

# Series: K ordenados ASC (menor K primero) + Pablo
series_k = [{'name': f"K={r['K']} · ζ≈{r['zeta']:.3f}", 'u': np.asarray(r['u']), 'K': r['K']}
            for r in sorted(results, key=lambda r: r['K'])]
series_p = {'name': f"Pablo (ρ·θ) · ζ≈{zeta_th:.3f}", 'u': np.asarray(u_th), 'K': None}

# Longitudes coherentes
N = len(z_axis)
for s in series_k + [series_p]:
    assert len(s['u']) == N, f"Longitud de u distinta a z_axis para {s['name']}."

# Paso vertical: que no se superpongan
Amax = float(np.nanmax([np.nanmax(np.abs(s['u'])) for s in (series_k + [series_p])]))
pad_factor = 0.15
step = (2.0 * Amax) * (1.0 + pad_factor) if Amax > 0 else 1.0

n_k = len(series_k)
# Offsets: Pablo arriba del todo, luego K con menor K más arriba
offset_pablo = step * (n_k + 1)
offsets_k = step * np.arange(n_k, 0, -1)  # n_k*step (arriba) -> 1*step (abajo)

# Colores
cmap = plt.get_cmap('tab10')
colors_k = [cmap(i % 10) for i in range(n_k)]
color_p = 'black'

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Pablo (arriba)
ax.plot(z_axis, series_p['u'] + offset_pablo, lw=1.4, color=color_p, label=series_p['name'])
ax.axhline(offset_pablo, color='0.85', lw=0.8, zorder=0)

# K (menor K arriba → mayor offset)
for (s, off, col) in zip(series_k, offsets_k, colors_k):
    ax.plot(z_axis, s['u'] + off, lw=1.2, color=col, label=s['name'])
    ax.axhline(off, color='0.85', lw=0.8, zorder=0)

# Ticks Y con etiquetas en cada línea base (Pablo primero)
yticks = [offset_pablo] + list(offsets_k)
yticklabels = [series_p['name']] + [s['name'] for s in series_k]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, fontsize=9)

ax.set_xlabel("z = ρ₀ θ  [px]")
ax.set_title("Parametrizaciones u(z) — todas en un único plot (Pablo arriba; menor K arriba)")
ax.grid(True, axis='x', alpha=0.25)

ax.set_xlim(z_axis.min(), z_axis.max())
ax.set_ylim(step*0.5, offset_pablo + 0.5*step)

plt.tight_layout()
plt.show()
