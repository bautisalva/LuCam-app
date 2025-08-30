
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, square
from skimage.measure import perimeter
from matplotlib.animation import FuncAnimation

# -----------------------------
# Función para generar círculos ruidosos
# -----------------------------
def noisy_circle_mask(size, r_base, noise_amp=5.0, n_points=1024, seed=None):
    if seed is not None:
        np.random.seed(seed)
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size//2, size//2
    dx, dy = x-cx, y-cy
    r_grid = np.hypot(dx, dy)
    theta_1d = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    f = np.sin(3*theta_1d + np.random.uniform(0, 2*np.pi))
    f += 0.5*np.sin(7*theta_1d + np.random.uniform(0, 2*np.pi))
    f += 0.3*np.sin(13*theta_1d + np.random.uniform(0, 2*np.pi))
    f /= np.max(np.abs(f))
    r_profile = r_base + noise_amp * f
    r_profile_ext = np.concatenate([r_profile, [r_profile[0]]])
    theta_ext = np.concatenate([theta_1d, [2*np.pi]])
    theta_grid = np.arctan2(dy, dx) % (2*np.pi)
    r_interp = np.interp(theta_grid.ravel(), theta_ext, r_profile_ext).reshape(theta_grid.shape)
    mask = (r_grid <= r_interp).astype(np.uint8)
    return mask, r_interp, theta_grid

# -----------------------------
# Función para calcular fajas conectadas al borde inicial
# -----------------------------
def decompose_bands_from_edge(mask_initial, mask_final, selem=None):
    if selem is None:
        selem = square(3)
    growth = (mask_final==1) & (mask_initial==0)
    shrink = (mask_initial==1) & (mask_final==0)
    
    # Bordes iniciales
    edge_initial = mask_initial.astype(bool) & (~binary_erosion(mask_initial, selem))
    
    # Crecimiento desde borde inicial
    layers_growth = []
    canvas = edge_initial.copy()
    remaining = growth.copy()
    while remaining.any():
        new_layer = remaining & binary_dilation(canvas, selem)
        if not new_layer.any():
            break
        layers_growth.append(new_layer.copy())
        canvas |= new_layer
        remaining &= ~new_layer
    
    # Retracción hacia borde inicial (opuesto)
    layers_shrink = []
    canvas_shrink = edge_initial.copy()
    remaining_shrink = shrink.copy()
    while remaining_shrink.any():
        new_layer = remaining_shrink & binary_dilation(canvas_shrink, selem)
        if not new_layer.any():
            break
        layers_shrink.append(new_layer.copy())
        canvas_shrink |= new_layer
        remaining_shrink &= ~new_layer
    
    # Combinar
    layers_all = layers_growth + [edge_initial] + layers_shrink[::-1]  # borde inicial en medio
    thickness = np.zeros_like(mask_initial, dtype=int)
    for i, layer in enumerate(layers_all):
        thickness[layer] = i+1
    return thickness, layers_all

# -----------------------------
# Perfil u(theta) y Var(u)
# -----------------------------
def u_theta_from_thickness(thickness, theta_grid, n_bins=360, stat='mean'):
    mask = thickness != 0
    bins = np.linspace(0, 2*np.pi, n_bins+1)
    centers = (bins[:-1]+bins[1:])/2
    u = np.zeros(n_bins, dtype=float)
    if not np.any(mask):
        return centers, u
    idx = np.digitize(theta_grid[mask], bins)-1
    idx = np.clip(idx, 0, n_bins-1)
    vals_all = thickness[mask]
    for k in range(n_bins):
        sel = (idx==k)
        if np.any(sel):
            u[k] = vals_all[sel].mean() if stat=='mean' else vals_all[sel].max()
    return centers, u

def var_from_profile(u, P=None):
    """Varianza ponderada por perímetro P si se desea"""
    u = np.asarray(u, dtype=float)
    return np.mean(u**2) - np.mean(u)**2

# -----------------------------
# Parámetros
# -----------------------------
SIZE = 512
R1 = 100
R2 = 140
NOISE1 = 12
NOISE2 = 15
SELEM = square(3)
NBINS = 360

# -----------------------------
# Generar dominios
# -----------------------------
mask1, r1_grid, theta_grid = noisy_circle_mask(SIZE, R1, NOISE1, seed=1)
mask2, r2_grid, _ = noisy_circle_mask(SIZE, R2, NOISE2, seed=2)

# -----------------------------
# Fajas correctamente desde borde inicial
# -----------------------------
thickness, layers = decompose_bands_from_edge(mask1, mask2, SELEM)

# -----------------------------
# u(theta) y Var(u)
# -----------------------------
centers, u_direct = u_theta_from_thickness(thickness, theta_grid, n_bins=NBINS)
var_fajas = var_from_profile(u_direct)

# -----------------------------
# Var(u) circular típica
# -----------------------------
r1_mean = np.mean(r1_grid, axis=0)
r2_mean = np.mean(r2_grid, axis=0)
xp = np.linspace(0, 2*np.pi, r1_mean.shape[0], endpoint=False)
r1_at_centers = np.interp(centers, xp, r1_mean)
r2_at_centers = np.interp(centers, xp, r2_mean)
u_circle = r2_at_centers - r1_at_centers
var_circle = var_from_profile(u_circle)

# -----------------------------
# Animación fajas
# -----------------------------
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(np.zeros_like(mask1), cmap='viridis', vmin=0, vmax=len(layers))
ax.set_title("Fajas desde borde inicial")
ax.axis('off')

def update(frame):
    canvas = np.zeros_like(mask1)
    for i in range(frame+1):
        canvas[layers[i]] = i+1
    im.set_data(canvas)
    return [im]

ani = FuncAnimation(fig, update, frames=len(layers), interval=300, blit=True)
ani.save("fajas_animacion.gif", writer="ffmpeg", fps=3)

plt.show()

# -----------------------------
# Resultados finales
# -----------------------------
print(f"Var(u) - método fajas: {var_fajas:.4f}")
print(f"Var(u) - círculo típico: {var_circle:.4f}")

# Perfil u(theta)
plt.figure(figsize=(8,4))
plt.plot(centers, u_direct, label="u(theta) - fajas")
plt.plot(centers, u_circle, '--', label="u(theta) - círculos")
plt.xlabel("θ [rad]")
plt.ylabel("Desplazamiento")
plt.title("Perfil u(theta)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Perímetro del dominio inicial
# -----------------------------
P1 = perimeter(mask1)
print(f"Perímetro del dominio inicial P1 = {P1:.2f}")

#%%

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, square
from skimage.measure import perimeter

# -----------------------------
# Función para generar círculos ruidosos
# -----------------------------
def noisy_circle_mask(size, r_base, noise_amp=5.0, n_points=1024, seed=None):
    if seed is not None:
        np.random.seed(seed)
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size//2, size//2
    dx, dy = x-cx, y-cy
    r_grid = np.hypot(dx, dy)
    theta_1d = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    f = np.sin(3*theta_1d + np.random.uniform(0, 2*np.pi))
    f += 0.5*np.sin(7*theta_1d + np.random.uniform(0, 2*np.pi))
    f += 0.3*np.sin(13*theta_1d + np.random.uniform(0, 2*np.pi))
    f /= np.max(np.abs(f))
    r_profile = r_base + noise_amp * f
    r_profile_ext = np.concatenate([r_profile, [r_profile[0]]])
    theta_ext = np.concatenate([theta_1d, [2*np.pi]])
    theta_grid = np.arctan2(dy, dx) % (2*np.pi)
    r_interp = np.interp(theta_grid.ravel(), theta_ext, r_profile_ext).reshape(theta_grid.shape)
    mask = (r_grid <= r_interp).astype(np.uint8)
    return mask, r_interp, theta_grid

# -----------------------------
# Fajas desde borde inicial
# -----------------------------
def decompose_bands_from_edge(mask_initial, mask_final, selem=None):
    if selem is None:
        selem = square(3)
    growth = (mask_final==1) & (mask_initial==0)
    shrink = (mask_initial==1) & (mask_final==0)
    
    edge_initial = mask_initial.astype(bool) & (~binary_erosion(mask_initial, selem))
    
    # Crecimiento
    layers_growth = []
    canvas = edge_initial.copy()
    remaining = growth.copy()
    while remaining.any():
        new_layer = remaining & binary_dilation(canvas, selem)
        if not new_layer.any():
            break
        layers_growth.append(new_layer.copy())
        canvas |= new_layer
        remaining &= ~new_layer
    
    # Retracción
    layers_shrink = []
    canvas_shrink = edge_initial.copy()
    remaining_shrink = shrink.copy()
    while remaining_shrink.any():
        new_layer = remaining_shrink & binary_dilation(canvas_shrink, selem)
        if not new_layer.any():
            break
        layers_shrink.append(new_layer.copy())
        canvas_shrink |= new_layer
        remaining_shrink &= ~new_layer
    
    layers_all = layers_growth + [edge_initial] + layers_shrink[::-1]
    thickness = np.zeros_like(mask_initial, dtype=int)
    for i, layer in enumerate(layers_all):
        thickness[layer] = i+1
    return thickness, layers_all

# -----------------------------
# Perfil u(theta) y Var(u)
# -----------------------------
def u_theta_from_thickness(thickness, theta_grid, n_bins=360, stat='mean'):
    mask = thickness != 0
    bins = np.linspace(0, 2*np.pi, n_bins+1)
    centers = (bins[:-1]+bins[1:])/2
    u = np.zeros(n_bins, dtype=float)
    if not np.any(mask):
        return centers, u
    idx = np.digitize(theta_grid[mask], bins)-1
    idx = np.clip(idx, 0, n_bins-1)
    vals_all = thickness[mask]
    for k in range(n_bins):
        sel = (idx==k)
        if np.any(sel):
            u[k] = vals_all[sel].mean() if stat=='mean' else vals_all[sel].max()
    return centers, u

def var_from_profile(u):
    u = np.asarray(u, dtype=float)
    return np.mean(u**2) - np.mean(u)**2

# -----------------------------
# Parámetros
# -----------------------------
SIZE = 512
R1 = 100
NOISE1 = 12
NOISE2 = 15
SELEM = square(3)
NBINS = 360

# -----------------------------
# Dominio inicial
# -----------------------------
mask1, r1_grid, theta_grid = noisy_circle_mask(SIZE, R1, NOISE1, seed=1)
P1 = perimeter(mask1)

# -----------------------------
# Diferentes desplazamientos
# -----------------------------
radii_final = np.linspace(R1+5, R1+60, 12)
var_fajas_list = []
var_circle_list = []

for R2 in radii_final:
    mask2, r2_grid, _ = noisy_circle_mask(SIZE, R2, NOISE2, seed=2)
    
    # Fajas
    thickness, layers = decompose_bands_from_edge(mask1, mask2, SELEM)
    centers, u_direct = u_theta_from_thickness(thickness, theta_grid, n_bins=NBINS)
    var_fajas_list.append(var_from_profile(u_direct))
    
    # Círculos típicos
    r1_mean = np.mean(r1_grid, axis=0)
    r2_mean = np.mean(r2_grid, axis=0)
    xp = np.linspace(0, 2*np.pi, r1_mean.shape[0], endpoint=False)
    r1_at_centers = np.interp(centers, xp, r1_mean)
    r2_at_centers = np.interp(centers, xp, r2_mean)
    u_circle = r2_at_centers - r1_at_centers
    var_circle_list.append(var_from_profile(u_circle))

# -----------------------------
# Graficar Var(u) vs desplazamiento
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(radii_final-R1, var_fajas_list, 'o-', label='Var(u) método fajas')
plt.plot(radii_final-R1, var_circle_list, 's--', label='Var(u) círculos típicos')
plt.xlabel("Separación de radios (final - inicial)")
plt.ylabel("Var(u)")
plt.title("Varianza del desplazamiento vs separación de dominios")
plt.legend()
plt.grid(True)
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, square
from skimage.measure import perimeter, find_contours
from scipy import ndimage
from matplotlib.animation import FuncAnimation

# -----------------------------
# Función para generar círculos ruidosos (sin cambios)
# -----------------------------
def noisy_circle_mask(size, r_base, noise_amp=5.0, n_points=1024, seed=None):
    if seed is not None:
        np.random.seed(seed)
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size//2, size//2
    dx, dy = x-cx, y-cy
    r_grid = np.hypot(dx, dy)
    theta_1d = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    f = np.sin(3*theta_1d + np.random.uniform(0, 2*np.pi))
    f += 0.5*np.sin(7*theta_1d + np.random.uniform(0, 2*np.pi))
    f += 0.3*np.sin(13*theta_1d + np.random.uniform(0, 2*np.pi))
    f /= np.max(np.abs(f))
    r_profile = r_base + noise_amp * f
    r_profile_ext = np.concatenate([r_profile, [r_profile[0]]])
    theta_ext = np.concatenate([theta_1d, [2*np.pi]])
    theta_grid = np.arctan2(dy, dx) % (2*np.pi)
    r_interp = np.interp(theta_grid.ravel(), theta_ext, r_profile_ext).reshape(theta_grid.shape)
    mask = (r_grid <= r_interp).astype(np.uint8)
    return mask, r_interp, theta_grid, r_profile, theta_1d

# -----------------------------
# NUEVO: Método mejorado usando transformada de distancia
# -----------------------------
def calculate_true_displacement(mask_initial, mask_final):
    # Calcular transformada de distancia desde el borde inicial
    edge_initial = mask_initial.astype(bool) & (~binary_erosion(mask_initial, square(3)))
    distance_transform = ndimage.distance_transform_edt(~edge_initial)
    
    # Asignar signos: positivo para crecimiento, negativo para retracción
    growth_region = (mask_final == 1) & (mask_initial == 0)
    shrink_region = (mask_initial == 1) & (mask_final == 0)
    
    displacement = np.zeros_like(mask_initial, dtype=float)
    displacement[growth_region] = distance_transform[growth_region]
    displacement[shrink_region] = -distance_transform[shrink_region]
    
    return displacement

# -----------------------------
# Función para calcular el perfil u(theta) desde el desplazamiento real
# -----------------------------
def u_theta_from_displacement(displacement, theta_grid, n_bins=360):
    mask = displacement != 0
    bins = np.linspace(0, 2*np.pi, n_bins+1)
    centers = (bins[:-1] + bins[1:]) / 2
    
    # Para cada bin angular, calcular el desplazamiento promedio
    u = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        in_bin = (theta_grid >= bins[i]) & (theta_grid < bins[i+1]) & mask
        if np.any(in_bin):
            u[i] = np.mean(displacement[in_bin])
            counts[i] = np.sum(in_bin)
    
    return centers, u, counts

# -----------------------------
# Función para calcular varianza (sin cambios)
# -----------------------------
def var_from_profile(u, P=None):
    u = np.asarray(u, dtype=float)
    return np.mean(u**2) - np.mean(u)**2

# -----------------------------
# Parámetros
# -----------------------------
SIZE = 512
R1 = 100
R2 = 140
NOISE1 = 12
NOISE2 = 15
NBINS = 360

# -----------------------------
# Generar dominios
# -----------------------------
mask1, r1_grid, theta_grid, r_profile1, theta_1d = noisy_circle_mask(SIZE, R1, NOISE1, seed=1)
mask2, r2_grid, _, r_profile2, _ = noisy_circle_mask(SIZE, R2, NOISE2, seed=2)

# -----------------------------
# Calcular desplazamiento real usando transformada de distancia
# -----------------------------
displacement = calculate_true_displacement(mask1, mask2)

# -----------------------------
# Calcular perfil u(theta) y varianza
# -----------------------------
centers, u_true, counts = u_theta_from_displacement(displacement, theta_grid, n_bins=NBINS)
var_true = var_from_profile(u_true)

# -----------------------------
# Comparar con método de círculos (mejorado)
# -----------------------------
# Interpolar los perfiles a los mismos ángulos
r1_interp = np.interp(centers, theta_1d, r_profile1)
r2_interp = np.interp(centers, theta_1d, r_profile2)
u_circle = r2_interp - r1_interp
var_circle = var_from_profile(u_circle)

# -----------------------------
# Visualización
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Mostrar máscaras
axes[0, 0].imshow(mask1, cmap='gray', alpha=0.5)
axes[0, 0].imshow(mask2, cmap='jet', alpha=0.5)
axes[0, 0].set_title("Máscaras superpuestas")
axes[0, 0].axis('off')

# Mostrar campo de desplazamiento
im = axes[0, 1].imshow(displacement, cmap='coolwarm')
plt.colorbar(im, ax=axes[0, 1])
axes[0, 1].set_title("Campo de desplazamiento")
axes[0, 1].axis('off')

# Comparar perfiles
axes[1, 0].plot(centers, u_true, 'b-', label='Desplazamiento real')
axes[1, 0].plot(centers, u_circle, 'r--', label='Diferencia de radios')
axes[1, 0].set_xlabel("θ [rad]")
axes[1, 0].set_ylabel("Desplazamiento")
axes[1, 0].set_title("Comparación de perfiles de desplazamiento")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Mostrar histograma de conteos por bin
axes[1, 1].bar(centers, counts, width=2*np.pi/NBINS, alpha=0.7)
axes[1, 1].set_xlabel("θ [rad]")
axes[1, 1].set_ylabel("Número de píxeles")
axes[1, 1].set_title("Distribución de píxeles por ángulo")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# Resultados finales
# -----------------------------
print(f"Var(u) - método transformada de distancia: {var_true:.4f}")
print(f"Var(u) - diferencia de radios: {var_circle:.4f}")
print(f"Diferencia relativa: {abs(var_true-var_circle)/var_true*100:.2f}%")

# -----------------------------
# Análisis de correlación entre métodos
# -----------------------------
correlation = np.corrcoef(u_true, u_circle)[0, 1]
print(f"Correlación entre los perfiles: {correlation:.4f}")

# Calcular error cuadrático medio
mse = np.mean((u_true - u_circle)**2)
print(f"Error cuadrático medio: {mse:.4f}")

#%%

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, square
from skimage.measure import find_contours
from scipy import ndimage, interpolate

# -----------------------------
# Función para generar círculos ruidosos (mejorada)
# -----------------------------
def noisy_circle_mask(size, r_base, noise_amp=5.0, n_points=1024, seed=None):
    if seed is not None:
        np.random.seed(seed)
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size//2, size//2
    dx, dy = x-cx, y-cy
    r_grid = np.hypot(dx, dy)
    theta_1d = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Generar ruido más controlado
    f = np.sin(3*theta_1d + np.random.uniform(0, 2*np.pi))
    f += 0.5*np.sin(7*theta_1d + np.random.uniform(0, 2*np.pi))
    f += 0.3*np.sin(13*theta_1d + np.random.uniform(0, 2*np.pi))
    f /= np.max(np.abs(f))
    
    r_profile = r_base + noise_amp * f
    r_profile_ext = np.concatenate([r_profile, [r_profile[0]]])
    theta_ext = np.concatenate([theta_1d, [2*np.pi]])
    theta_grid = np.arctan2(dy, dx) % (2*np.pi)
    r_interp = np.interp(theta_grid.ravel(), theta_ext, r_profile_ext).reshape(theta_grid.shape)
    mask = (r_grid <= r_interp).astype(np.uint8)
    return mask, r_interp, theta_grid, r_profile, theta_1d

# -----------------------------
# Método mejorado para calcular desplazamiento normal
# -----------------------------
def calculate_normal_displacement(mask_initial, mask_final, center):
    # Encontrar contornos
    contour_initial = find_contours(mask_initial, 0.5)[0]
    contour_final = find_contours(mask_final, 0.5)[0]
    
    # Convertir a coordenadas polares centradas
    y_i, x_i = contour_initial[:, 0], contour_initial[:, 1]
    y_f, x_f = contour_final[:, 0], contour_final[:, 1]
    
    dx_i, dy_i = x_i - center[1], y_i - center[0]
    dx_f, dy_f = x_f - center[1], y_f - center[0]
    
    r_i = np.hypot(dx_i, dy_i)
    r_f = np.hypot(dx_f, dy_f)
    theta_i = np.arctan2(dy_i, dx_i) % (2*np.pi)
    theta_f = np.arctan2(dy_f, dx_f) % (2*np.pi)
    
    # Interpolar para obtener valores en los mismos ángulos
    theta_common = np.linspace(0, 2*np.pi, 360, endpoint=False)
    
    # Ordenar por ángulo para interpolación
    sort_idx_i = np.argsort(theta_i)
    sort_idx_f = np.argsort(theta_f)
    
    r_i_sorted = r_i[sort_idx_i]
    theta_i_sorted = theta_i[sort_idx_i]
    r_f_sorted = r_f[sort_idx_f]
    theta_f_sorted = theta_f[sort_idx_f]
    
    # Interpolar
    r_i_interp = interpolate.interp1d(theta_i_sorted, r_i_sorted, 
                                     bounds_error=False, fill_value="extrapolate")(theta_common)
    r_f_interp = interpolate.interp1d(theta_f_sorted, r_f_sorted, 
                                     bounds_error=False, fill_value="extrapolate")(theta_common)
    
    # Calcular desplazamiento normal (diferencia de radios)
    normal_disp = r_f_interp - r_i_interp
    
    return theta_common, normal_disp, r_i_interp, r_f_interp

# -----------------------------
# Parámetros
# -----------------------------
SIZE = 512
R1 = 100
R2 = 140
NOISE1 = 12
NOISE2 = 100
CENTER = (SIZE//2, SIZE//2)
NBINS = 360

# -----------------------------
# Generar dominios
# -----------------------------
mask1, r1_grid, theta_grid, r_profile1, theta_1d = noisy_circle_mask(SIZE, R1, NOISE1, seed=1)
mask2, r2_grid, _, r_profile2, _ = noisy_circle_mask(SIZE, R2, NOISE2, seed=2)

# -----------------------------
# Calcular desplazamiento normal
# -----------------------------
theta_common, normal_disp, r_i_interp, r_f_interp = calculate_normal_displacement(mask1, mask2, CENTER)

# -----------------------------
# Calcular varianza
# -----------------------------
var_normal = np.var(normal_disp)

# -----------------------------
# Comparar con método anterior de diferencia de radios
# -----------------------------
r1_interp = np.interp(theta_common, theta_1d, r_profile1)
r2_interp = np.interp(theta_common, theta_1d, r_profile2)
radial_disp = r2_interp - r1_interp
var_radial = np.var(radial_disp)

# -----------------------------
# Visualización
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Contornos superpuestos
axes[0, 0].imshow(mask1, cmap='Greys', alpha=0.3)
axes[0, 0].imshow(mask2, cmap='Blues', alpha=0.3)
contour1 = find_contours(mask1, 0.5)[0]
contour2 = find_contours(mask2, 0.5)[0]
axes[0, 0].plot(contour1[:, 1], contour1[:, 0], 'r-', linewidth=2, label='Contorno inicial')
axes[0, 0].plot(contour2[:, 1], contour2[:, 0], 'b-', linewidth=2, label='Contorno final')
axes[0, 0].set_title("Contornos superpuestos")
axes[0, 0].legend()
axes[0, 0].axis('off')

# Perfiles de radio
axes[0, 1].plot(theta_common, r_i_interp, 'r-', label='Radio inicial')
axes[0, 1].plot(theta_common, r_f_interp, 'b-', label='Radio final')
axes[0, 1].set_xlabel("Ángulo [rad]")
axes[0, 1].set_ylabel("Radio [píxeles]")
axes[0, 1].set_title("Perfiles de radio")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Comparación de desplazamientos
axes[1, 0].plot(theta_common, normal_disp, 'g-', label='Desplazamiento normal')
axes[1, 0].plot(theta_common, radial_disp, 'm--', label='Diferencia de radios')
axes[1, 0].set_xlabel("Ángulo [rad]")
axes[1, 0].set_ylabel("Desplazamiento [píxeles]")
axes[1, 0].set_title("Comparación de métodos de desplazamiento")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Histograma de diferencias
diff = normal_disp - radial_disp
axes[1, 1].hist(diff, bins=30, alpha=0.7, color='orange')
axes[1, 1].axvline(np.mean(diff), color='red', linestyle='--', label=f'Media: {np.mean(diff):.2f}')
axes[1, 1].set_xlabel("Diferencia entre métodos")
axes[1, 1].set_ylabel("Frecuencia")
axes[1, 1].set_title("Distribución de diferencias")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# Resultados
# -----------------------------
print(f"Varianza (desplazamiento normal): {var_normal:.4f}")
print(f"Varianza (diferencia de radios): {var_radial:.4f}")
print(f"Diferencia relativa: {abs(var_normal-var_radial)/var_normal*100:.2f}%")

# Calcular métricas de similitud
correlation = np.corrcoef(normal_disp, radial_disp)[0, 1]
mse = np.mean((normal_disp - radial_disp)**2)
print(f"Correlación: {correlation:.4f}")
print(f"Error cuadrático medio: {mse:.4f}")

# -----------------------------
# Análisis adicional: suavizado de perfiles
# -----------------------------
from scipy import signal

# Suavizar los perfiles con un filtro de media móvil
window_size = 5
window = np.ones(window_size) / window_size

r_i_smooth = np.convolve(r_i_interp, window, mode='same')
r_f_smooth = np.convolve(r_f_interp, window, mode='same')
normal_disp_smooth = r_f_smooth - r_i_smooth

var_normal_smooth = np.var(normal_disp_smooth)

print(f"\nDespués de suavizado (ventana={window_size}):")
print(f"Varianza (desplazamiento normal suavizado): {var_normal_smooth:.4f}")
print(f"Diferencia con método original: {abs(var_normal-var_normal_smooth)/var_normal*100:.2f}%")