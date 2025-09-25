import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, filters, morphology, util, color, io
from skimage.draw import polygon
from scipy.ndimage import distance_transform_edt
import os, glob
from skimage.measure import find_contours


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
    
BASE_DIR = r"F:\Pablo Domenichini - Medidas2\2021\P8137\DC\140"
BASENAME = "Bin-P8137-140Oe-5ms-"
IDX      = 5
EXTS     = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]

path = buscar_imagen(BASE_DIR, BASENAME, IDX, EXTS)
im = cargar_binaria(path)

#%%
def _fft_lowpass_closed(curve, K_keep):
    z = curve[:,0] + 1j*curve[:,1]
    Z = np.fft.fft(z); M = len(z)
    keep = np.zeros(M, dtype=bool); keep[0] = True
    for k in range(1, K_keep+1):
        keep[k%M] = True; keep[-k%M] = True
    z_s = np.fft.ifft(np.where(keep, Z, 0))
    return np.column_stack([z_s.real, z_s.imag])


C = find_contours(im, level=0.5)
contour = max(C, key=len)  

smooth = _fft_lowpass_closed(contour, K_keep=2)

plt.figure(figsize=(8, 8))
plt.imshow(im, cmap='gray')
plt.plot(contour[:, 1], contour[:, 0], 'r--', label='Original')
plt.plot(smooth[:, 1], smooth[:, 0], 'b-', label='Suavizado')
plt.legend()
plt.axis('equal')
plt.title('Contorno original vs suavizado (FFT)')
plt.show()

#%%

def get_normals_from_fft(curve, K_keep):
    z = curve[:, 0] + 1j * curve[:, 1]
    M = len(z)
    
    # Obtener la FFT y truncar
    Z = np.fft.fft(z)
    freqs = np.fft.fftfreq(M)  # frecuencias normalizadas
    keep = np.zeros(M, dtype=bool); keep[0] = True
    for k in range(1, K_keep+1):
        keep[k % M] = True
        keep[-k % M] = True
    Z_trunc = np.where(keep, Z, 0)
    
    # Derivada de z(t) -> dz/dt ≈ derivada en Fourier
    dZ = Z_trunc * (2j * np.pi * freqs)  # derivada espectral

    # Evaluar la serie inversa para obtener dz/dt
    dz_dt = np.fft.ifft(dZ)

    # Normales: rotar 90 grados (multiplicar por i)
    normals_complex = 1j * dz_dt

    # Separar en componentes x, y
    normals = np.column_stack((normals_complex.real, normals_complex.imag))

    # Normalizar vectores
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_unit = normals / (norms + 1e-10)  # para evitar división por cero

    return normals_unit

# Normales
normals = get_normals_from_fft(contour, K_keep=2)

# Mostrar imagen con contorno y normales
plt.figure(figsize=(8, 8))
plt.imshow(im, cmap='gray')
plt.plot(smooth[:,1], smooth[:,0], 'b-', label='Contorno suavizado')

# Dibujar normales (reducidas en tamaño para visualización)
scale = 10
for p, n in zip(smooth[::10], normals[::10]):  # cada 10 puntos para claridad
    plt.arrow(p[1], p[0], n[1]*scale, n[0]*scale, 
              head_width=1, head_length=1.5, fc='r', ec='r')

plt.axis('equal')
plt.legend()
plt.title('Normales sobre contorno suavizado')
plt.show()

#%%

from scipy.spatial import cKDTree

def compute_normal_distance_field(smooth, normals, contour_real, max_dist=50, num_steps=100):
    """
    Calcula u(s): distancia desde el contorno suave al real a lo largo de la normal.
    - smooth: contorno suavizado (Nx2)
    - normals: vectores normales unitarios en cada punto de smooth
    - contour_real: contorno original binario (Mx2)
    - max_dist: cuánto extender la línea normal en ambas direcciones
    - num_steps: resolución de muestreo en cada normal
    """
    tree = cKDTree(contour_real)
    u_values = []
    L_total = np.sum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))
    s_vals = [0]  # acumulado de longitud

    for i, (p, n) in enumerate(zip(smooth, normals)):
        # Construir línea de muestreo a lo largo de la normal
        ts = np.linspace(-max_dist, max_dist, num_steps)
        line_points = p + ts[:, None] * n[None, :]

        # Buscar punto más cercano en el contorno real
        dists, _ = tree.query(line_points)
        idx_min = np.argmin(dists)
        u = ts[idx_min]  # distancia firmada (puede ser negativa)

        u_values.append(u)

        # Calcular s (longitud acumulada)
        if i > 0:
            ds = np.linalg.norm(smooth[i] - smooth[i - 1])
            s_vals.append(s_vals[-1] + ds)

    return np.array(s_vals), np.array(u_values)

# Asumiendo que ya tenemos: smooth, normals, contour_real
s_vals, u_vals = compute_normal_distance_field(smooth, normals, contour)

# Graficar campo escalar u(s)
plt.figure(figsize=(10, 4))
plt.plot(s_vals, u_vals, label='u(s): distancia firmada')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Longitud de arco (s)')
plt.ylabel('Distancia firmada u(s)')
plt.title('Campo escalar a lo largo del contorno suave')
plt.legend()
plt.grid(True)
plt.show()

#%%

import numpy as np
from scipy.spatial import cKDTree

# --- util: quitar el punto duplicado del cierre si existe
def ensure_closed_no_dup(C):
    if np.allclose(C[0], C[-1]):
        return C[:-1].copy()
    return C.copy()

# --- util: reparametrización por longitud de arco a N puntos
def resample_by_arclength(C, N):
    C = ensure_closed_no_dup(C)
    # acumulado de arco
    d = np.linalg.norm(np.diff(C, axis=0, append=C[:1]), axis=1)
    s = np.concatenate(([0.0], np.cumsum(d)))  # s in [0, L]
    L = s[-1]
    # objetivo: N puntos equiespaciados
    s_target = np.linspace(0, L, N, endpoint=False)
    # interp lineal por tramo periódico
    x = np.interp(s_target, s, np.r_[C[:,0], C[0,0]])
    y = np.interp(s_target, s, np.r_[C[:,1], C[0,1]])
    return np.column_stack([x, y])

# --- FFT low-pass con opción de oversampling por zero-padding
def fft_lowpass_closed_equal_samples(curve_eq, K_keep, M_out=None):
    """
    curve_eq: Nx2 equiespaciada por arco.
    K_keep: nº de armónicos a conservar (±k).
    M_out: opcional, nº de puntos en reconstrucción (oversampling). Si None, usa N.
    """
    z = curve_eq[:,0] + 1j*curve_eq[:,1]
    Z = np.fft.fft(z); N = len(z)
    keep = np.zeros(N, dtype=bool); keep[0] = True
    for k in range(1, K_keep+1):
        keep[k % N] = True
        keep[-k % N] = True
    Z_lp = np.where(keep, Z, 0)

    if M_out is None or M_out == N:
        z_s = np.fft.ifft(Z_lp)
    else:
        # zero-padding en frecuencia para oversampling espacial
        # reordenamos a modo "fftshift" manual para colocar bajas f en el centro
        Z_shift = np.fft.fftshift(Z_lp)
        pad = (M_out - N)
        pad_left = pad // 2
        pad_right = pad - pad_left
        Z_pad = np.pad(Z_shift, (pad_left, pad_right), mode='constant')
        Z_pad = np.fft.ifftshift(Z_pad)
        z_s = np.fft.ifft(Z_pad) * (M_out / N)  # factor de escala por padding
    return np.column_stack([z_s.real, z_s.imag])

# --- Derivada espectral correcta y normales unitarias (dz/ds)
def normals_from_fft(curve_eq, K_keep, M_out=None):
    # 1) low-pass (y oversampling opcional)
    smooth = fft_lowpass_closed_equal_samples(curve_eq, K_keep, M_out)
    z = smooth[:,0] + 1j*smooth[:,1]
    M = len(z)

    # 2) FFT y derivada con frecuencias enteras k
    Z = np.fft.fft(z)
    freqs = np.fft.fftfreq(M, d=1.0/M)  # -> enteros k
    dZ = Z * (2j * np.pi * freqs)       # dz/dt con t in [0,1)
    dz_dt = np.fft.ifft(dZ)

    # 3) arclength speed |dz/dt| y tangente unitaria
    speed = np.abs(dz_dt) + 1e-15
    T = (dz_dt / speed)  # tangente unitaria (complex)

    # 4) normal unitaria = rotación +90°: i*T
    N = 1j * T
    normals = np.column_stack([N.real, N.imag])
    smooth_xy = np.column_stack([z.real, z.imag])
    return smooth_xy, normals

# --- Intersección de rayo p + t*n con polilínea (contour_real)
def ray_polyline_intersections(p, n, poly):
    """
    p: punto (2,)
    n: dirección normal unitaria (2,)
    poly: Nx2 (cerrado o no). Trabajamos con segmentos [i -> i+1].
    Retorna: lista de t (t>=0 hacia n; t<=0 si fuese hacia -n)
    """
    ts = []
    # aseguramos polígono cerrado en lazo
    P = ensure_closed_no_dup(poly)
    for i in range(len(P)):
        a = P[i]
        b = P[(i+1) % len(P)]
        ab = b - a
        # Resolver p + t n = a + u ab
        # 2x2: [n, -ab] [t, u]^T = a - p
        A = np.column_stack([n, -ab])
        rhs = a - p
        det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
        if np.abs(det) < 1e-12:
            continue  # paralelos o casi
        invA = (1.0/det) * np.array([[ A[1,1], -A[0,1]],
                                     [-A[1,0],  A[0,0]]])
        t, u = invA @ rhs
        # intersección sobre el segmento si u in [0,1]
        if (u >= -1e-12) and (u <= 1+1e-12):
            ts.append(t)
    return ts

def compute_u_along_normals_by_intersection(smooth, normals, contour_real, max_search=200.0):
    """
    Para cada punto p del contorno suave, busca la intersección más cercana
    del rayo p + t n con el contorno real (polilínea). Devuelve t (firmado).
    Si no encuentra intersección en [−max_search, +max_search], usa fallback NN.
    """
    # fallback NN continuo (proyección a segmento)
    tree = cKDTree(contour_real)
    u_vals = []
    s_vals = [0.0]
    for i, (p, n) in enumerate(zip(smooth, normals)):
        ts = ray_polyline_intersections(p, n, contour_real)

        # elegimos el t con |t| mínimo (y |t| <= max_search)
        t_pick = None
        if ts:
            ts = [t for t in ts if np.abs(t) <= max_search]
            if ts:
                t_pick = ts[np.argmin(np.abs(ts))]

        if t_pick is None:
            # Fallback: NN sobre una recta densa (pero ahora continuo + chico)
            # Tomamos un muestreo fino SOLO para el fallback.
            ts_fine = np.linspace(-max_search, max_search, 801)
            line_points = p + ts_fine[:, None] * n[None, :]
            dists, _ = tree.query(line_points)
            t_pick = ts_fine[np.argmin(dists)]

        u_vals.append(t_pick)

        if i > 0:
            ds = np.linalg.norm(smooth[i] - smooth[i - 1])
            s_vals.append(s_vals[-1] + ds)
    return np.array(s_vals), np.array(u_vals)

#%%
    
# 1) Asegurar contorno más largo y uniforme
contour = max(C, key=len)
contour = ensure_closed_no_dup(contour)
# cuidado: find_contours da [fila, col] => [y, x], si querés mantener tu convención, está ok.

# 2) Reparametrizar por arco (por ej. N=2048)
contour_eq = resample_by_arclength(contour, N=1024)

# 3) Curva suave + normales (con oversampling a 4096 si querés más “continuidad”)
smooth, normals = normals_from_fft(contour_eq, K_keep=2, M_out=1024)

# 4) u(s) por intersección exacta con la polilínea del contorno real
s_vals, u_vals = compute_u_along_normals_by_intersection(smooth, normals, contour, max_search=150.0)

# 5) Graficar
plt.figure(figsize=(10, 4))
plt.plot(s_vals, u_vals, lw=1.5)
plt.axhline(0, color='k', ls='--', lw=0.8)
plt.xlabel('s (long. de arco)')
plt.ylabel('u(s) (px, firmado)')
plt.title('Distancia a lo largo de la normal (intersección exacta)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%

# ==== Plots con fondo blanco, contornos coloreados y distancias en rojo ====
# Requiere: im, contour, smooth, normals, s_vals, u_vals

import numpy as np
import matplotlib.pyplot as plt

# Parámetros
every = 1          # dibujar 1 de cada 'every' segmentos
lw_cnt = 1.6
lw_seg = 1.2
ms_pts = 4.5

# 1) u(s): distancia firmada vs. longitud de arco
plt.figure(figsize=(10, 4), facecolor='white')
plt.plot(s_vals, u_vals, '-', lw=1.5, color='black')
plt.axhline(0, color='gray', ls='--', lw=0.9)
plt.xlabel('s (longitud de arco)')
plt.ylabel('u(s) [px]')
plt.title('Distancia firmada punto a punto sobre la normal')
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# 2) Contornos (original y suavizado) + segmentos de distancia (rojo) sobre fondo blanco
# Nota: los contornos están en convención (fila=y, col=x)
pints = smooth + (u_vals[:, None] * normals)   # p_int = p + u n

fig = plt.figure(figsize=(8, 8), facecolor='white')
ax  = fig.add_subplot(111)

# Contorno original (color 1)
ax.plot(contour[:, 1], contour[:, 0], color='#2E86C1', lw=lw_cnt, label='Contorno original')

# Contorno suavizado (color 2)
ax.plot(smooth[:, 1],  smooth[:, 0],  color='#16A085', lw=lw_cnt, label='Contorno suavizado')

# Segmentos de distancia en rojo
idx = np.arange(len(smooth))[::every]
for i in idx:
    y0, x0 = smooth[i, 0], smooth[i, 1]
    y1, x1 = pints[i, 0],  pints[i, 1]
    ax.plot([x0, x1], [y0, y1], color='red', lw=lw_seg)


ax.set_aspect('equal', adjustable='box')
ax.invert_yaxis()  # para mantener la orientación "de imagen" si tus datos vienen de find_contours
ax.set_title('Contornos y distancias sobre fondo blanco')
ax.legend(loc='lower right', frameon=False)
plt.tight_layout()
plt.show()

# 3) Histograma de |u|
plt.figure(figsize=(6, 4), facecolor='white')
plt.hist(np.abs(u_vals), bins=40, edgecolor='black')
plt.xlabel('|u| [px]')
plt.ylabel('Cuenta')
plt.title('Distribución de magnitudes de distancia')
plt.tight_layout()
plt.show()

#%%

# ==== Visualizar exactamente las normales "Fourier-derivar y rotar por i" ====
# Requiere en el entorno: contour, _fft_lowpass_closed, get_normals_from_fft
# (opcional) smooth si querés comparar; no es necesario.

import numpy as np
import matplotlib.pyplot as plt

# Parámetros de visualización
K_keep_plot = 2      # usá el mismo K que venís usando para las normales
every = 15           # dibujar 1 de cada 'every' flechas
L = 18.0             # longitud visual de la flecha en px (solo para el segmento auxiliar)

# 1) Curva reconstruida con los mismos armónicos (esta es la base sobre la que están definidas las normales)
curve_fourier = _fft_lowpass_closed(contour, K_keep=K_keep_plot)

# 2) Normales que salen de derivar la serie truncada y rotar por i (tu función)
normals_fft = get_normals_from_fft(contour, K_keep=K_keep_plot)  # misma M y fase que curve_fourier

# 3) Plot: contorno Fourier + normales (solo estas)
fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')

# (opcional) contorno real, solo como referencia
ax.plot(contour[:, 1], contour[:, 0], color='#2E86C1', lw=1.1, alpha=0.35, label='Contorno real')

# curva base de la serie truncada (sobre esta están definidas las normales)
ax.plot(curve_fourier[:, 1], curve_fourier[:, 0], color='#16A085', lw=1.8, label='Serie Fourier (K_keep={})'.format(K_keep_plot))

# flechas de normales (derivar + rotar i)
idx = np.arange(len(curve_fourier))[::every]
ax.quiver(curve_fourier[idx, 1], curve_fourier[idx, 0],
          normals_fft[idx, 1], normals_fft[idx, 0],
          angles='xy', scale_units='xy', scale=1.0,
          width=0.004, headwidth=3.5, headlength=4.5, headaxislength=4.5,
          color='#E67E22', label='Normal (FFT derivada)',
          pivot='mid', minlength=0)

# además, trazo segmentos rectos para ver la dirección en L px
for i in idx:
    y0, x0 = curve_fourier[i, 0], curve_fourier[i, 1]
    ny, nx = normals_fft[i, 0], normals_fft[i, 1]
    ax.plot([x0, x0 + L*nx], [y0, y0 + L*ny], color='#E67E22', lw=1.0, alpha=0.8)

ax.set_aspect('equal', adjustable='box')
ax.invert_yaxis()  # para mantener la convención de imagen (fila=y)
ax.set_title('Normales de la serie de Fourier truncada (derivar + rotar por i)')
ax.legend(loc='lower right', frameon=False)
plt.tight_layout()
plt.show()
