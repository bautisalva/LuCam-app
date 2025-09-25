
"""
RUGOSIDAD — FOURIER + FAJAS (script final con logs y overlay estilo ECG)

Produce 5 gráficos:
  1) Chequeo de normales (fajas coloreadas por u) para K=2.
  2) B(r) para K = 1..10 y 20, todas en un gráfico con ζ en la leyenda.
  3) ζ vs K para el conjunto {1..10, 20}.
  4) ζ vs K para un barrido fino (linspace) sin mostrar B(r).
  5) u(z) sobre eje común z = ρ0 θ, en modo "electrocardiograma":
     Pablo arriba, luego K=1, K=2, K=3 y K=20 (una debajo de otra, con offsets).

Además:
  - Ordena outputs en out/YYYYmmdd-HHMMSS_rugosidad/
  - Guarda resultados en .npz y parámetros en meta.json
  - Mensajes en consola para seguimiento.

Autor: vos + tu bot nerd.
"""

from __future__ import annotations
import os, glob, json, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from skimage import io, color, util, filters, morphology, measure
from skimage.draw import polygon
from scipy.ndimage import distance_transform_edt

# ================================
# CONFIGURACIÓN (EDITABLE)
# ================================
CFG = dict(
    # --- Entrada (elige una):
    IMAGE_PATH=None,       # path directo a la imagen binaria o en escala de grises
    BASE_DIR=r"C:\Users\Marina\Documents\Labo 6\LuCam-app\analisis\rugosidad",
    BASENAME="Bin-P8137-150Oe-3ms-",    # ejemplo: "Bin-P8137-150Oe-3ms-"
    IDX=5,                 # índice para el patrón
    EXTS=(".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"),

    # --- Parámetros geométricos y de análisis:
    MSAMPLES=1024,             # puntos por arclonga (nuestro método)
    FIT_RANGE_LOG10=(0, 0.8),  # ventana para el ajuste en log10(r)
    DRAW_EVERY=1,              # fajas a dibujar (1=todas) para el plot de normales

    # --- Barrido discreto (para plots 2 y 3):
    K_SET_BASE=list(range(1, 11)) + [20],  # {1..10} ∪ {20}

    # --- Barrido fino (plot 4, ζ vs K sin B(r)):
    K_FINE_MIN=1,
    K_FINE_MAX=80,
    K_FINE_N=64,               # cantidad de puntos en linspace

    # --- Comparación en eje común (plot 5):
    K_OVERLAY=[1, 2, 3, 20],   # los K a superponer con Pablo

    # --- Salida:
    ROOT_OUT="out",
    TAG="rugosidad",
    SHOW_PLOTS=False           # True para ver en pantalla además de guardar
)

# ================================
# UTILIDADES DE ENTRADA/SALIDA
# ================================
def ensure_outdir(root="out", tag="rugosidad"):
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(root) / f"{ts}_{tag}"
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    return outdir

def save_fig(fig, outdir, name, dpi=220):
    p = Path(outdir) / "figs" / f"{name}.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    print(f"[fig guardada] {p}")

def save_npz(outdir, name, **arrays):
    p = Path(outdir) / f"{name}.npz"
    np.savez_compressed(p, **arrays)
    print(f"[npz guardado] {p}")

def save_meta(outdir, **meta):
    p = Path(outdir) / "meta.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[json guardado] {p}")

def buscar_imagen(base_dir, basename, idx, exts):
    for ext in exts:
        p = os.path.join(base_dir, f"{basename}{idx}{ext}")
        if os.path.exists(p): return p
    cand = glob.glob(os.path.join(base_dir, f"{basename}{idx}*"))
    cand = [p for p in cand if os.path.splitext(p)[1].lower() in exts]
    return cand[0] if cand else ""

# ================================
# CARGA / BINARIZACIÓN
# ================================
def cargar_binaria(path, min_area=64):
    """Carga imagen, convierte a gris si hace falta, umbraliza (Otsu) y limpia."""
    print("[info] Cargando imagen y binarizando…")
    im = io.imread(path)
    if im.ndim == 3 and im.shape[-1] == 4:
        im = im[..., :3]
    if im.ndim == 3:
        im = color.rgb2gray(im)
    im = util.img_as_float(im)
    thr = filters.threshold_otsu(im)
    mask = im >= thr
    mask = morphology.remove_small_holes(mask, area_threshold=min_area)
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    print("[ok] Máscara binaria lista.")
    return mask

# ================================
# GEOMETRÍA DE CURVAS
# ================================
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

def extract_largest_contour(mask):
    print("[info] Extrayendo contorno principal…")
    conts = measure.find_contours(mask.astype(float), 0.5)
    if not conts:
        raise ValueError("No se encontró contorno.")
    conts_xy = [c[:, ::-1] for c in conts]  # (x,y)
    conts_xy = [c if _polygon_area(c) > 0 else np.flipud(c) for c in conts_xy]
    perims = [ _perimeter(c) for c in conts_xy ]
    print("[ok] Contorno principal extraído.")
    return conts_xy[int(np.argmax(perims))]

def resample_closed(poly, M):
    """Remuestrea curva cerrada por arclonga a M puntos (excluye duplicado final)."""
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

def fft_lowpass_closed(curve, K_keep):
    """Suaviza curva cerrada reteniendo k=0 y ±1..±K_keep."""
    z = curve[:,0] + 1j*curve[:,1]
    Z = np.fft.fft(z); M = len(Z)
    keep = np.zeros(M, bool); keep[0] = True
    kmax = min(K_keep, M//2)
    for k in range(1, kmax+1):
        keep[k] = True; keep[-k] = True
    z_s = np.fft.ifft(np.where(keep, Z, 0))
    return np.column_stack([z_s.real, z_s.imag])

def rasterize_closed(curve, shape):
    C = _ensure_closed(curve)[:-1]
    rr, cc = polygon(C[:,1], C[:,0], shape=shape)  # (row=y, col=x)
    m = np.zeros(shape, dtype=bool); m[rr, cc] = True
    return m

# ================================
# DISTANCIA FIRMADA Y NORMALES
# ================================
def signed_distance_and_grad(mask):
    d_in  = distance_transform_edt(mask)
    d_out = distance_transform_edt(~mask)
    d = d_out.astype(np.float32); d[mask] = -d_in[mask].astype(np.float32)
    gy, gx = np.gradient(d)  # grad(d) = (gx, gy) en coords (x,y)
    return d, gx, gy

def bilinear(arr, x, y):
    H, W = arr.shape
    x0 = int(np.floor(x)); x1 = x0 + 1
    y0 = int(np.floor(y)); y1 = y0 + 1
    x0c = np.clip(x0, 0, W-1); x1c = np.clip(x1, 0, W-1)
    y0c = np.clip(y0, 0, H-1); y1c = np.clip(y1, 0, H-1)
    wx = x - x0; wy = y - y0
    v00 = arr[y0c, x0c]; v10 = arr[y0c, x1c]
    v01 = arr[y1c, x0c]; v11 = arr[y1c, x1c]
    return (1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + (1-wx)*wy*v01 + wx*wy*v11

# ================================
# INTERSECCIÓN RAYO–POLÍGONO
# ================================
def ray_segment_intersection(p, d, a, b, eps=1e-12):
    """Intersección rayo p + t d (t>=0) con segmento [a,b]. Devuelve (t,u) ó (inf,None)."""
    v = b - a
    M = np.array([[d[0], -v[0]], [d[1], -v[1]]], dtype=float)
    rhs = (a - p).astype(float)
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    if abs(det) < eps:
        return np.inf, None
    inv = (1.0/det) * np.array([[ M[1,1], -M[0,1]], [-M[1,0], M[0,0]]], float)
    t, u = (inv @ rhs)
    if t >= -1e-12 and (0.0 - 1e-12 <= u <= 1.0 + 1e-12):
        return max(0.0, t), u
    return np.inf, None

def ray_polygon_nearest_intersection(p, n_hat, poly):
    P = _ensure_closed(poly)
    best_t = np.inf; best_q = p
    for a, b in zip(P[:-1], P[1:]):
        t, u = ray_segment_intersection(p, n_hat, a, b)
        if t < best_t:
            best_t = t
            best_q = a + u*(b-a) if u is not None else p
    if not np.isfinite(best_t): return np.nan, p
    return float(best_t), best_q

def fajas_u_Q(C_ref, C_real, N_dir):
    M = len(C_ref)
    u = np.zeros(M, float); Q = np.zeros_like(C_ref)
    for j in range(M):
        p = C_ref[j]; n = N_dir[j]
        tpos, qpos = ray_polygon_nearest_intersection(p,  n, C_real)
        tneg, qneg = ray_polygon_nearest_intersection(p, -n, C_real)
        cand = []
        if np.isfinite(tpos): cand.append((+tpos, qpos, +1))
        if np.isfinite(tneg): cand.append((-tneg, qneg, -1))
        if cand:
            cand.sort(key=lambda c: abs(c[0]))
            u[j], Q[j] = cand[0][0], cand[0][1]
        else:
            u[j] = 0.0; Q[j] = p
    return u, Q

# ================================
# RUGOSIDAD B(r) Y AJUSTES
# ================================
def compute_B_of_r(u, L):
    """B(r_m) = ⟨[u(z+mΔs)-u(z)]²⟩, Δs=L/M, r_m=mΔs, m=1..M-1 (envuelto)."""
    u = np.asarray(u, float)
    M = len(u); ds = L / M
    r = np.arange(1, M, dtype=float) * ds
    B = np.empty_like(r)
    idx = np.arange(M)
    for m in range(1, M):
        diff = u[(idx+m)%M] - u
        B[m-1] = np.mean(diff*diff)
    return r, B

def fit_loglog(x, y, log10_range):
    m = (x>0) & (y>0) & np.isfinite(x) & np.isfinite(y)
    lx, ly = np.log10(x[m]), np.log10(y[m])
    if len(lx) < 3:
        return np.nan, np.nan
    lo, hi = log10_range
    sel = (lx >= lo) & (lx <= hi)
    if sel.sum() < 2:
        sel = np.ones_like(lx, bool)
    p = np.polyfit(lx[sel], ly[sel], 1)
    slope = float(p[0]); zeta = 0.5 * slope
    return zeta, slope

# ================================
# MÉTODO DE PABLO (ρ·θ)
# ================================
def contorno_principal(mask):
    conts = measure.find_contours(mask.astype(float), 0.5)
    if not conts: return None
    C = max(conts, key=lambda c: c.shape[0])  # (y,x)
    if np.linalg.norm(C[0]-C[-1]) > 1.0:
        C = np.vstack([C, C[0]])
    y, x = C[:,0], C[:,1]
    return x, y

def polygon_centroid(x, y):
    if np.hypot(x[0]-x[-1], y[0]-y[-1]) < 1e-9:
        x, y = x[:-1], y[:-1]
    X, Y = np.r_[x, x[0]], np.r_[y, y[0]]
    cross = X[:-1]*Y[1:] - X[1:]*Y[:-1]
    A = 0.5*np.sum(cross)
    if abs(A) < 1e-12: return float(np.mean(x)), float(np.mean(y))
    Cx = (1/(6*A))*np.sum((X[:-1]+X[1:])*cross)
    Cy = (1/(6*A))*np.sum((Y[:-1]+Y[1:])*cross)
    return float(Cx), float(Cy)

def resample_xy_arclength(x, y, N):
    if np.hypot(x[0]-x[-1], y[0]-y[-1]) < 1e-9:
        x, y = x[:-1], y[:-1]
    seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    s   = np.r_[0.0, np.cumsum(seg[:-1])]
    L   = s[-1] + seg[-1]
    t   = s / L
    te  = np.r_[t, 1.0]; xe = np.r_[x, x[0]]; ye = np.r_[y, y[0]]
    tg  = np.linspace(0, 1, N, endpoint=False)
    xs  = np.interp(tg, te, xe); ys = np.interp(tg, te, ye)
    return xs, ys, L

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

# ================================
# PLOTS
# ================================
def plot_fajas(C_real, C_ref, Q, u, title, draw_every=1):
    fig, ax = plt.subplots(1,1, figsize=(6.8,6.8))
    ax.plot(C_real[:,0], C_real[:,1], 'k-', lw=1, alpha=0.6, label='contorno real')
    ax.plot(C_ref[:,0],  C_ref[:,1],  '-',  lw=2, color='tab:blue', label='curva suave')
    norm = TwoSlopeNorm(vcenter=0.0, vmin=np.nanmin(u), vmax=np.nanmax(u))
    cmap = plt.get_cmap('coolwarm')
    for j in range(0, len(u), draw_every):
        if not np.isfinite(u[j]): continue
        col = cmap(norm(u[j]))
        ax.plot([C_ref[j,0], Q[j,0]], [C_ref[j,1], Q[j,1]], '-', lw=1.0, color=col)
    ax.set_aspect('equal'); ax.set_title(title); ax.legend(loc='lower right', fontsize=8)
    sc = ax.scatter([np.nan],[np.nan], c=[0], cmap=cmap, norm=norm)
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label('u [px]')
    plt.tight_layout()
    return fig

# ================================
# CORE DEL MÉTODO (NUESTRO)
# ================================
def u_B_por_K(mask, Msamples, K, fit_range_log10):
    """Devuelve: dict con C_real, C_ref, u, Q, r, B, zeta, slope, L."""
    C0 = extract_largest_contour(mask)            # (x,y)
    C_real, L_real = resample_closed(C0, Msamples)
    C_ref = fft_lowpass_closed(C_real, K)
    L_ref = _perimeter(C_ref)

    H, W = mask.shape
    M_ref = rasterize_closed(C_ref, (H, W))
    _, gx, gy = signed_distance_and_grad(M_ref)

    N_dir = np.zeros_like(C_ref)
    for j, p in enumerate(C_ref):
        g = np.array([bilinear(gx, p[0], p[1]), bilinear(gy, p[0], p[1])], float)
        nv = np.linalg.norm(g)
        if nv < 1e-8:
            j0 = (j-1) % Msamples; j1 = (j+1) % Msamples
            t = C_ref[j1] - C_ref[j0]
            t /= (np.linalg.norm(t) + 1e-12)
            N_dir[j] = np.array([t[1], -t[0]])
        else:
            N_dir[j] = g / nv

    u, Q = fajas_u_Q(C_ref, C_real, N_dir)
    r, B = compute_B_of_r(u, L_ref)
    zeta, slope = fit_loglog(r, B, fit_range_log10)
    return dict(C_real=C_real, C_ref=C_ref, Q=Q, u=u, r=r, B=B, zeta=zeta, slope=slope, L=L_ref)

# ================================
# OVERLAY EN EJE COMÚN (z = ρ0 θ) + ECG
# ================================
def overlay_curves_common_axis_ecg(mask, Msamples, K_list, N_TH, fit_range_log10):
    """
    Devuelve:
      z_axis (ρ0θ), u_pablo, {u_K_on_common for K in K_list}, zeta_pablo (opcional).
    Además calcula offsets y etiquetas para ploteo estilo ECG.
    """
    print("[info] Preparando overlay en eje común (z = ρ0 θ)…")

    x_raw, y_raw = contorno_principal(mask)
    xc, yc = polygon_centroid(x_raw, y_raw)
    i_anchor = int(np.argmax(x_raw - xc))
    theta_anchor = float(np.mod(np.arctan2(y_raw[i_anchor]-yc, x_raw[i_anchor]-xc), 2*np.pi))

    print("[info]  • Trazando rayos angulares para Pablo…")
    theta_th, rtheta = raycast_r_of_theta(x_raw, y_raw, xc, yc, Ntheta=N_TH, theta0=theta_anchor)
    rho0 = float(np.nanmean(rtheta))
    u_th = rtheta - rho0
    dtheta = (2*np.pi)/N_TH
    dz = rho0 * dtheta
    z_axis = np.arange(N_TH, dtype=float) * dz

    # Nuestro método (referencia por Fourier) → normales → u(z)
    print("[info]  • Remuestreo por arclonga del contorno real…")
    xs, ys, _ = resample_xy_arclength(x_raw, y_raw, Msamples)
    z_real = xs + 1j*ys

    def fft_lowpass_complex(z, K):
        Z = np.fft.fft(z); N = Z.size
        Zf = np.zeros_like(Z)
        kmax = min(K, N//2)
        Zf[:kmax+1] = Z[:kmax+1]
        if kmax>0: Zf[-kmax:] = Z[-kmax:]
        return np.fft.ifft(Zf)

    H, W = mask.shape
    uK_on_common = {}

    for K in K_list:
        print(f"[info]  • Procesando overlay para K={K}…")
        z_ref = fft_lowpass_complex(z_real, K)
        x_ref, y_ref = np.real(z_ref), np.imag(z_ref)

        M_ref = rasterize_closed(np.column_stack([x_ref, y_ref]), (H, W))
        _, gx, gy = signed_distance_and_grad(M_ref)
        N_dir = np.zeros((Msamples, 2), float)
        for j in range(Msamples):
            g = np.array([bilinear(gx, x_ref[j], y_ref[j]),
                          bilinear(gy, x_ref[j], y_ref[j])], float)
            nv = np.linalg.norm(g)
            if nv < 1e-8:
                j0 = (j-1) % Msamples; j1 = (j+1) % Msamples
                t = np.array([x_ref[j1]-x_ref[j0], y_ref[j1]-y_ref[j0]])
                t /= (np.linalg.norm(t) + 1e-12)
                N_dir[j] = np.array([t[1], -t[0]])
            else:
                N_dir[j] = g / nv

        u, _ = fajas_u_Q(np.column_stack([x_ref, y_ref]), np.column_stack([xs, ys]), N_dir)

        # Mapear esta referencia a z = ρ0 θ con el mismo ancla
        theta_k = np.unwrap(np.arctan2(y_ref - yc, x_ref - xc))
        z_k = rho0 * (theta_k - theta_anchor)
        P_ref = 2*np.pi*rho0
        z_k_wrapped = np.mod(z_k, P_ref)
        idx = np.argsort(z_k_wrapped)
        z_sorted = z_k_wrapped[idx]; u_sorted = u[idx]
        z_ext = np.r_[z_sorted, z_sorted[0] + P_ref]
        u_ext = np.r_[u_sorted, u_sorted[0]]
        u_on_common = np.interp(z_axis, z_ext, u_ext)
        uK_on_common[K] = u_on_common

    # ζ de Pablo (opcional)
    def compute_B_wrap(u):
        u = np.asarray(u, float); N = u.size
        B = np.empty(N-1, float); idx = np.arange(N)
        for k in range(1, N):
            diff = u[(idx+k)%N] - u
            B[k-1] = np.mean(diff*diff)
        return B

    r_th = np.arange(1, len(u_th), dtype=float) * dz
    B_th = compute_B_wrap(u_th)
    zeta_th, _ = fit_loglog(r_th, B_th, fit_range_log10)

    print("[ok] Overlay preparado.")
    return dict(z_axis=z_axis, rho0=rho0, dz=dz, u_pablo=u_th, uK=uK_on_common, zeta_pablo=zeta_th)

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    print("[inicio] Script de rugosidad (Fourier + Fajas)")

    outdir = ensure_outdir(CFG["ROOT_OUT"], CFG["TAG"])

    # Resolver path de entrada
    print("[info] Resolviendo imagen de entrada…")
    if CFG["IMAGE_PATH"] is None:
        path = buscar_imagen(CFG["BASE_DIR"], CFG["BASENAME"], CFG["IDX"], CFG["EXTS"])
        if not path:
            raise FileNotFoundError("No encontré la imagen. Ajustá BASE_DIR/BASENAME/IDX/EXTS o IMAGE_PATH.")
    else:
        path = CFG["IMAGE_PATH"]
    print(f"[input] {path}")

    mask = cargar_binaria(path)

    # ------------------------------------------------------------
    # (1) Chequeo de normales (K=2): fajas coloreadas por u
    # ------------------------------------------------------------
    print("[tarea 1] Chequeo de normales con K=2…")
    K_norm = 2
    res_norm = u_B_por_K(mask, Msamples=CFG["MSAMPLES"], K=K_norm, fit_range_log10=CFG["FIT_RANGE_LOG10"])
    figA = plot_fajas(res_norm["C_real"], res_norm["C_ref"], res_norm["Q"], res_norm["u"],
                      title=f"Chequeo de normales (fajas) — K={K_norm}", draw_every=CFG["DRAW_EVERY"])
    save_fig(figA, outdir, "A_fajas_K2")
    if not CFG["SHOW_PLOTS"]: plt.close(figA)

    # ------------------------------------------------------------
    # (2) B(r) para K in {1..10, 20} con ζ en la leyenda
    # ------------------------------------------------------------
    print("[tarea 2] Curvas B(r) para K en {1..10, 20}…")
    K_list_base = CFG["K_SET_BASE"]
    results_base = []
    figB = plt.figure(figsize=(8.6,5.8))
    for K in K_list_base:
        print(f"[info]  • Calculando B(r) para K={K}…")
        res = u_B_por_K(mask, CFG["MSAMPLES"], K, CFG["FIT_RANGE_LOG10"])
        results_base.append(dict(K=K, r=res["r"], B=res["B"], zeta=res["zeta"], slope=res["slope"]))
        plt.loglog(res["r"], res["B"], '.', ms=3.0, alpha=0.9, label=f"K={K} · ζ≈{res['zeta']:.3f}")
    plt.xlabel("r (arclonga) [px]"); plt.ylabel(r"B(r) [px$^2$]")
    plt.title("B(r) — K = 1..10 y 20 (nuestro método)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    save_fig(figB, outdir, "B_Br_multi_K_1a10_y_20")
    if not CFG["SHOW_PLOTS"]: plt.close(figB)

    # ------------------------------------------------------------
    # (3) ζ vs K para el conjunto {1..10, 20}
    # ------------------------------------------------------------
    print("[tarea 3] Rugosidad ζ vs K (K en {1..10, 20})…")
    Ks_base = [d["K"] for d in results_base]
    Zs_base = [d["zeta"] for d in results_base]
    figC = plt.figure(figsize=(6.8,4.8))
    plt.plot(Ks_base, Zs_base, 'o-', lw=1.6)
    plt.xlabel("K (modos mantenidos)"); plt.ylabel("ζ (rugosidad)")
    plt.title("ζ vs K — conjunto {1..10, 20}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(figC, outdir, "C_zeta_vs_K_base")
    if not CFG["SHOW_PLOTS"]: plt.close(figC)

    # ------------------------------------------------------------
    # (4) ζ vs K para un linspace fino (sin mostrar B(r))
    # ------------------------------------------------------------
    print("[tarea 4] Barrido fino de K (ζ vs K) sin mostrar B(r)…")
    K_fine_vals = np.unique(np.linspace(CFG["K_FINE_MIN"], CFG["K_FINE_MAX"], CFG["K_FINE_N"], dtype=int))
    Zs_fine = []
    for K in K_fine_vals:
        print(f"[info]  • ζ(K): procesando K={K}…")
        res = u_B_por_K(mask, CFG["MSAMPLES"], K, CFG["FIT_RANGE_LOG10"])
        Zs_fine.append(res["zeta"])
    Zs_fine = np.array(Zs_fine, float)
    figD = plt.figure(figsize=(7.6,4.8))
    plt.plot(K_fine_vals, Zs_fine, '-', lw=1.6, marker='.', ms=5)
    plt.xlabel("K (modos mantenidos)")
    plt.ylabel("ζ (rugosidad)")
    plt.title(f"ζ vs K — barrido fino [{CFG['K_FINE_MIN']}, {CFG['K_FINE_MAX']}] (N={len(K_fine_vals)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(figD, outdir, "D_zeta_vs_K_fino")
    if not CFG["SHOW_PLOTS"]: plt.close(figD)

    # ------------------------------------------------------------
    # (5) u(z) en eje común z = ρ0 θ, estilo ECG (Pablo arriba, luego K=1,2,3,20)
    # ------------------------------------------------------------
    print("[tarea 5] Overlay de u(z) en eje común (ECG): Pablo + K=1,2,3,20…")
    N_TH = CFG["MSAMPLES"]  # resolución angular comparable al muestreo por arclonga
    overlay = overlay_curves_common_axis_ecg(
        mask=mask,
        Msamples=CFG["MSAMPLES"],
        K_list=CFG["K_OVERLAY"],
        N_TH=N_TH,
        fit_range_log10=CFG["FIT_RANGE_LOG10"]
    )
    z_axis = overlay["z_axis"]
    series = []
    # Arriba Pablo
    series.append(("Pablo (ρ·θ)", overlay["u_pablo"]))
    # Luego K en el orden pedido
    for K in CFG["K_OVERLAY"]:
        series.append((f"K={K}", overlay["uK"][K]))

    # Estimar offset vertical por amplitud
    print("[info]  • Construyendo figura estilo ECG…")
    amps = [np.nanmax(np.abs(u)) if np.size(u)>0 else 1.0 for _, u in series]
    base_step = (2.0 * np.nanmax(amps)) * 1.15 if np.nanmax(amps) > 0 else 1.0
    offsets = [base_step * (len(series) - i) for i in range(1, len(series)+1)]  # Pablo más arriba

    figE = plt.figure(figsize=(10.2, 6.4))
    ax = plt.gca()
    for (label, u), off in zip(series, offsets):
        ax.plot(z_axis, u + off, lw=1.3, label=label)
        ax.axhline(off, color='0.85', lw=0.8, zorder=0)
    ax.set_xlabel("z = ρ₀ θ  [px]")
    ax.set_ylabel("u(z) + offset [px]")
    ax.set_title("u(z) — estilo ECG (Pablo arriba; luego K=1,2,3,20)")
    ax.set_xlim(z_axis.min(), z_axis.max())
    ax.set_ylim(min(offsets) - 0.5*base_step, max(offsets) + 0.5*base_step)
    ax.set_yticks(offsets)
    ax.set_yticklabels([lbl for lbl, _ in series], fontsize=9)
    ax.grid(True, axis='x', alpha=0.25)
    plt.tight_layout()
    save_fig(figE, outdir, "E_u_superpuestas_ECG")
    if not CFG["SHOW_PLOTS"]: plt.close(figE)

    # ------------------------------------------------------------
    # GUARDAR DATOS / META
    # ------------------------------------------------------------
    print("[info] Guardando paquetes de resultados…")
    save_npz(outdir, "resultados_base",
             Ks=np.array([d["K"] for d in results_base], int),
             zetas=np.array([d["zeta"] for d in results_base], float))
    save_npz(outdir, "resultados_fino",
             K_fine=np.array(K_fine_vals, int),
             zetas_fino=Zs_fine)
    # guardo también B(r) de los base para reproducibilidad
    pack_B = {f"r_K{d['K']}": d["r"] for d in results_base}
    pack_B.update({f"B_K{d['K']}": d["B"] for d in results_base})
    save_npz(outdir, "Br_curvas_base", **pack_B)
    # overlay ECG
    save_npz(outdir, "overlay_common_axis_ECG",
             z_axis=overlay["z_axis"], dz=overlay["dz"], rho0=overlay["rho0"],
             u_pablo=overlay["u_pablo"],
             **{f"uK_{K}": overlay["uK"][K] for K in CFG["K_OVERLAY"]})

    meta_to_save = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in CFG.items()}
    meta_to_save["input_path"] = path
    save_meta(outdir, **meta_to_save)

    if CFG["SHOW_PLOTS"]:
        plt.show()
    else:
        plt.close('all')

    print(f"[fin] Todo listo. Outputs en: {outdir}")
