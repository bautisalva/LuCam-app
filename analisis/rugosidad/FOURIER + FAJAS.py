# -*- coding: utf-8 -*-
# ============================================================
# Comparación limpia en una sola celda (con escala a micrómetros):
#   - B(r) (rugosidad) y S(q) (factor de estructura)
#   - Pablo (ρ·θ) vs tu método (normales por fajas + rayos) para varios K
#   - Escala: 0.4 µm / pixel
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, measure, morphology, filters, util
from skimage.draw import polygon
from scipy.ndimage import distance_transform_edt

# ------------------ Parámetros de entrada ------------------
BASE_DIR = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe\080"
BASENAME = "Bin-P8137-080Oe-100ms-"
IDX      = 5
EXTS     = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]

# Escala espacial
PIXEL_SIZE_UM = 0.4   # micrómetros por pixel

# Muestreo, Fourier y barrido K
N_SAMPLES   = 1024            # muestreo en arco (subí/bajá para velocidad)
K_LIST      = [1]   # K de la curva suave

# Rango de ajuste (en décadas log10) para B(r) y S(q)
FIT_B_RANGE  = (-0.7, 0.8)     # usa r ∈ [10^a, 10^b] (en µm)
FIT_SQ_RANGE = (-1, 1)    # usa q ∈ [10^a, 10^b] (en rad/µm)

# ------------------ Helpers generales ------------------
def buscar_imagen(base_dir, basename, idx, exts):
    import os, glob
    for ext in exts:
        p = os.path.join(base_dir, f"{basename}{idx}{ext}")
        if os.path.exists(p): return p
    cand = glob.glob(os.path.join(base_dir, f"{basename}{idx}*"))
    cand = [p for p in cand if os.path.splitext(p)[1].lower() in exts]
    return cand[0] if cand else ""

def cargar_binaria(path, thresh=0.5):
    """Carga imagen. Si es RGB, pasa a gris; binariza a bool usando umbral fijo u Otsu si es float."""
    im = io.imread(path)
    if im.ndim == 3 and im.shape[-1] == 4: im = im[..., :3]
    if im.ndim == 3:
        im = color.rgb2gray(im)  # [0,1]
        return (im > thresh).astype(bool)
    if np.issubdtype(im.dtype, np.floating):
        t = filters.threshold_otsu(im)
        return (im >= t).astype(bool)
    elif np.issubdtype(im.dtype, np.integer):
        return (im > (np.iinfo(im.dtype).max * thresh)).astype(bool)
    return (im > thresh).astype(bool)

def contorno_principal(mask):
    """Devuelve contorno exterior más grande en (x,y), CCW."""
    m = morphology.remove_small_holes(mask, area_threshold=64)
    m = morphology.remove_small_objects(m, min_size=64)
    conts = measure.find_contours(m.astype(float), 0.5)
    if not conts:
        raise RuntimeError("No se encontró contorno en la máscara.")
    conts_xy = [c[:, ::-1] for c in conts]  # (x,y)
    def poly_area(xy):
        x, y = xy[:,0], xy[:,1]
        return 0.5*np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
    conts_xy = [c if poly_area(c) > 0 else np.flipud(c) for c in conts_xy]
    def perim(xy): return np.sum(np.hypot(np.diff(xy[:,0], append=xy[0,0]),
                                          np.diff(xy[:,1], append=xy[0,1])))
    C = max(conts_xy, key=perim)
    return C[:,0], C[:,1]  # x, y

def centroid(x, y):
    X, Y = np.r_[x, x[0]], np.r_[y, y[0]]
    cross = X[:-1]*Y[1:] - X[1:]*Y[:-1]
    A = 0.5*np.sum(cross)
    if abs(A) < 1e-12:
        return float(np.mean(x)), float(np.mean(y))
    Cx = (1/(6*A))*np.sum((X[:-1]+X[1:])*cross)
    Cy = (1/(6*A))*np.sum((Y[:-1]+Y[1:])*cross)
    return float(Cx), float(Cy)

def resample_arclength(x, y, N, i_anchor=0):
    """Re-muestrea contorno cerrado en N puntos equiespaciados en arco, partiendo del índice ancla."""
    x = np.asarray(x); y = np.asarray(y)
    if np.hypot(x[0]-x[-1], y[0]-y[-1]) > 1e-9:
        x = np.r_[x, x[0]]; y = np.r_[y, y[0]]
    x = np.roll(x, -i_anchor); y = np.roll(y, -i_anchor)
    seg = np.hypot(np.diff(x), np.diff(y))
    s   = np.r_[0.0, np.cumsum(seg)]
    L   = s[-1]
    t   = s / L
    tgrid = np.linspace(0, 1, N, endpoint=False)
    xs = np.interp(tgrid, t, x); ys = np.interp(tgrid, t, y)
    return xs, ys, L

def fft_lowpass_complex(z, K):
    """Pasa-bajo de Fourier: conserva 0 y ±1..±K sobre señal compleja z."""
    if K <= 0: return z
    Z = np.fft.fft(z); N = Z.size
    Zf = np.zeros_like(Z)
    kmax = min(K, N//2)
    Zf[:kmax+1] = Z[:kmax+1]
    if kmax > 0: Zf[-kmax:] = Z[-kmax:]
    return np.fft.ifft(Zf)

def rasterize_polygon(x, y, shape):
    from skimage.draw import polygon as _poly
    rr, cc = _poly(y, x, shape=shape)
    m = np.zeros(shape, dtype=bool)
    m[rr, cc] = True
    return m

def signed_distance_and_grad(mask):
    """Distancia firmada (afuera +, adentro -) y su gradiente (gx,gy)."""
    d_in  = distance_transform_edt(mask)
    d_out = distance_transform_edt(~mask)
    d = d_out.astype(np.float32); d[mask] = -d_in[mask].astype(np.float32)
    gy, gx = np.gradient(d)  # np.gradient devuelve (d/dy, d/dx)
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

def ray_segment_intersection_one_side(p, n_hat, xR, yR, tmin=1e-6):
    """Interseca rayo p + t n_hat (t>tmin) con contorno poligonal (xR,yR). Devuelve (t_min, q)."""
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
        if t > tmin and 0.0 <= s <= 1.0 and t < best_t:
            best_t = t; best_q = p + t*n_hat
    if not np.isfinite(best_t): return np.nan, p
    return float(best_t), best_q

def u_by_normals_and_rays(x_ref, y_ref, x_real, y_real, gx, gy):
    """Para cada punto de la curva suave, normal = grad(d)/||grad(d)||, u por intersección ±normal."""
    N = len(x_ref)
    u  = np.zeros(N, float)
    qx = np.zeros(N, float); qy = np.zeros(N, float)
    for j in range(N):
        p = np.array([x_ref[j], y_ref[j]])
        g = np.array([bilinear_sample(gx, x_ref[j], y_ref[j]),
                      bilinear_sample(gy, x_ref[j], y_ref[j])], float)
        nv = np.linalg.norm(g)
        if nv < 1e-8:
            jm = (j-1) % N; jp = (j+1) % N
            t = np.array([x_ref[jp]-x_ref[jm], y_ref[jp]-y_ref[jm]])
            t /= (np.linalg.norm(t) + 1e-12)
            n = np.array([t[1], -t[0]])
        else:
            n = g / nv
        tpos, qpos = ray_segment_intersection_one_side(p,  n, x_real, y_real)
        tneg, qneg = ray_segment_intersection_one_side(p, -n, x_real, y_real)
        cand = []
        if np.isfinite(tpos): cand.append((+tpos, qpos))
        if np.isfinite(tneg): cand.append((-tneg, qneg))
        if cand:
            cand.sort(key=lambda c: abs(c[0]))
            u[j]  = cand[0][0]
            qx[j], qy[j] = cand[0][1]
        else:
            u[j]  = 0.0
            qx[j], qy[j] = p[0], p[1]
    return u, qx, qy

def compute_B_wrap(u, dz):
    """B(r) periódico: r = k*dz, k=1..N-1."""
    u = np.asarray(u, float); N = u.size
    idx = np.arange(N)
    B = np.array([np.mean((u[(idx+k)%N] - u)**2) for k in range(1, N)], float)
    r = np.arange(1, N, dtype=float) * dz
    return r, B

def fit_loglog(x, y, lohi_decades):
    """Ajuste lineal en log10 dentro de [10^a, 10^b]. Devuelve (slope, intercept, R2)."""
    a, b = lohi_decades
    m = (x > 0) & (y > 0) & (np.log10(x) >= a) & (np.log10(x) <= b)
    if np.count_nonzero(m) < 3: return np.nan, np.nan, np.nan
    lx, ly = np.log10(x[m]), np.log10(y[m])
    p = np.polyfit(lx, ly, 1)
    yhat = np.polyval(p, lx)
    ss_res = np.sum((ly - yhat)**2); ss_tot = np.sum((ly - ly.mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(p[0]), float(p[1]), float(R2)

def structure_factor(u, dz):
    """S(q) ∝ |Û(q)|^2 (relativa). Devuelve q>0 [rad/um] y S(q) correspondiente."""
    u = np.asarray(u, float)
    N = u.size
    u0 = u - np.nanmean(u)
    U = np.fft.rfft(u0)                  # 0..Nyquist
    S = (np.abs(U)**2) / N
    freqs = np.fft.rfftfreq(N, d=dz)     # [ciclos/µm]
    q = 2*np.pi*freqs                    # [rad/µm]
    return q[1:], S[1:]                  # descarto q=0

def raycast_r_of_theta(x, y, xc, yc, Ntheta, theta0=0.0, eps=1e-12):
    """Para Pablo: r(θ) por intersección de rayos desde el centroide, arrancando en θ0."""
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

# ------------------ Carga imagen y contorno ------------------
path = buscar_imagen(BASE_DIR, BASENAME, IDX, EXTS)
assert path, "No encontré la imagen. Ajustá BASE_DIR/BASENAME/IDX/EXTS."
mask = cargar_binaria(path)           # bool
x_raw, y_raw = contorno_principal(mask)
xc, yc = centroid(x_raw, y_raw)

# ------------------ Ancla común (mismo punto de partida) ------------------
i_anchor = int(np.argmax(x_raw - xc))  # punto más a la derecha respecto al centroide
theta_anchor = float(np.mod(np.arctan2(y_raw[i_anchor]-yc, x_raw[i_anchor]-xc), 2*np.pi))

# Contorno real remuestreado a N_SAMPLES desde el ancla (en PIXELES)
xR, yR, L_real_px = resample_arclength(x_raw, y_raw, N_SAMPLES, i_anchor=i_anchor)
zR = xR + 1j*yR

# ------------------ Pablo (ρ·θ): eje común z=ρ0 θ (escala aplicada) ------------------
theta_grid, rtheta_px = raycast_r_of_theta(x_raw, y_raw, xc, yc, Ntheta=N_SAMPLES, theta0=theta_anchor)
rho0_px = float(np.nanmean(rtheta_px))          # radio medio [px]
rho0_um = rho0_px * PIXEL_SIZE_UM               # [µm]
u_th_um = (rtheta_px - rho0_px) * PIXEL_SIZE_UM # u(θ) en [µm]
dtheta  = (2*np.pi)/N_SAMPLES
dz_um   = rho0_um * dtheta                      # paso del eje común [µm]
z_axis_um = np.arange(N_SAMPLES, dtype=float) * dz_um

# B(r) y S(q) para Pablo (en unidades físicas)
r_th_um, B_th_um = compute_B_wrap(u_th_um, dz_um)
q_th,     S_th   = structure_factor(u_th_um, dz_um)
sB_th, bB_th, R2B_th = fit_loglog(r_th_um, B_th_um, FIT_B_RANGE)
zeta_th = 0.5*sB_th if np.isfinite(sB_th) else np.nan
sS_th, bS_th, R2S_th = fit_loglog(q_th, S_th, FIT_SQ_RANGE)

# ------------------ Tu método para varios K (con escala) ------------------
H, W = mask.shape
results = []  # guardo por K: u(z) [µm], B(r) [µm²], q [rad/µm], S(q), slopes, etc.

for K in K_LIST:
    # 1) curva suave (pasa-bajo sobre contorno real remuestreado)
    z_ref  = fft_lowpass_complex(zR, K)
    x_ref  = np.real(z_ref); y_ref = np.imag(z_ref)

    # 2) fajas → normales: distancia firmada a la curva suave y su gradiente
    M_ref  = rasterize_polygon(x_ref, y_ref, (H, W))
    _, gx, gy = signed_distance_and_grad(M_ref)

    # 3) u en PIXELES por intersección de rayos ±normal con contorno real
    uK_px, qxK, qyK = u_by_normals_and_rays(x_ref, y_ref, xR, yR, gx, gy)

    # 4) mapear a eje común z=ρ0(θ-θ0) y regrillar a malla uniforme de Pablo (sin cambiar z)
    thetaK = np.unwrap(np.arctan2(y_ref - yc, x_ref - xc))
    zK_px  = rho0_px * (thetaK - theta_anchor)       # en [px]
    P_ref_px  = 2*np.pi*rho0_px
    zK_mod_px = np.mod(zK_px, P_ref_px)

    idx = np.argsort(zK_mod_px)
    z_sorted_px = zK_mod_px[idx]; u_sorted_px = uK_px[idx]
    z_ext_px = np.r_[z_sorted_px, z_sorted_px[0] + P_ref_px]
    u_ext_px = np.r_[u_sorted_px, u_sorted_px[0]]
    # interpolo en el eje z de Pablo pero en pixeles y luego convierto a µm:
    z_axis_px = z_axis_um / PIXEL_SIZE_UM
    u_on_common_px = np.interp(z_axis_px, z_ext_px, u_ext_px)

    # Convertir a micrómetros
    u_on_common_um = u_on_common_px * PIXEL_SIZE_UM

    # 5) B(r) y S(q) en unidades físicas
    rK_um, BK_um = compute_B_wrap(u_on_common_um, dz_um)
    qK,    SK    = structure_factor(u_on_common_um, dz_um)

    # 6) ajustes
    sB, bB, R2B = fit_loglog(rK_um, BK_um, FIT_B_RANGE)
    zeta = 0.5*sB if np.isfinite(sB) else np.nan
    sS, bS, R2S = fit_loglog(qK, SK, FIT_SQ_RANGE)

    results.append(dict(
        K=K, u=u_on_common_um, r=rK_um, B=BK_um, q=qK, S=SK,
        sB=sB, zeta=zeta, R2B=R2B, sS=sS, R2S=R2S
    ))

# ------------------ Gráfico único final: B(r) y S(q) ------------------
fig, (axB, axS) = plt.subplots(1, 2, figsize=(12.5, 5.0))

# Panel B(r): Pablo + todos los K (log–log)
axB.loglog(r_th_um, B_th_um, '.', ms=3.5, color='k',
           label=f"Pablo ρ·θ · ζ≈{zeta_th:.3f} (R²={R2B_th:.2f})")
for res in sorted(results, key=lambda d: d['K']):
    lbl = f"K={res['K']} · ζ≈{res['zeta']:.3f} (R²={res['R2B']:.2f})"
    axB.loglog(res['r'], res['B'], '.', ms=3, alpha=0.9, label=lbl)
axB.set_xlabel("r [µm]"); axB.set_ylabel(r"B(r) [µm$^2$]")
axB.set_title("Función de rugosidad B(r)")
axB.legend(ncol=2, fontsize=8)
axB.grid(True, which='both', alpha=0.2)

# Panel S(q): Pablo + todos los K (log–log)
axS.loglog(q_th, S_th, '-', lw=1.2, color='k',
           label=f"Pablo ρ·θ · slope≈{sS_th:.2f} (R²={R2S_th:.2f})")
for res in sorted(results, key=lambda d: d['K']):
    lbl = f"K={res['K']} · slope≈{res['sS']:.2f} (R²={res['R2S']:.2f})"
    axS.loglog(res['q'], res['S'], '-', lw=1.0, alpha=0.95, label=lbl)
axS.set_xlabel("q [rad/µm]"); axS.set_ylabel(r"S(q) [u.u.]")
axS.set_title("Factor de estructura S(q)")
axS.legend(ncol=2, fontsize=8)
axS.grid(True, which='both', alpha=0.2)

plt.tight_layout()
plt.show()

# ------------------ Resumen por consola ------------------
print(f"[Pablo]  ζ ≈ {zeta_th:.3f}  |  slope S(q) ≈ {sS_th:.2f}")
for res in sorted(results, key=lambda d: d['K']):
    print(f"[K={res['K']:>3}] ζ ≈ {res['zeta']:.3f}  |  slope S(q) ≈ {res['sS']:.2f}")
