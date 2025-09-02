
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



#%%

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, square
from matplotlib.animation import FuncAnimation

# -----------------------------
# Función para generar círculos ruidosos (igual que la tuya)
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
# Fajas conectadas al borde inicial
# (DEVUELVE capas para animar SIN el borde)
# -----------------------------
def decompose_bands_from_edge(mask_initial, mask_final, selem=None):
    if selem is None:
        selem = square(3)

    growth = (mask_final == 1) & (mask_initial == 0)
    shrink = (mask_initial == 1) & (mask_final == 0)

    # borde inicial (una sola vez, NO se animará)
    edge_initial = mask_initial.astype(bool) & (~binary_erosion(mask_initial, selem))

    # crecer pegado al borde
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

    # retraer pegado al borde
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

    # IMPORTANTE: para la ANIMACIÓN excluimos el borde
    layers_for_anim = layers_growth + layers_shrink[::-1]

    # thickness (por si lo querés usar en otro lado). Aquí incluimos el borde,
    # pero OJO: 'layers' que devuelve la función es SIN borde.
    layers_all_for_thickness = layers_growth + [edge_initial] + layers_shrink[::-1]
    thickness = np.zeros_like(mask_initial, dtype=int)
    for i, layer in enumerate(layers_all_for_thickness):
        thickness[layer] = i + 1

    return thickness, layers_for_anim, edge_initial

# -----------------------------
# Parámetros
# -----------------------------
SIZE  = 512
R1    = 100
R2    = 140
NOISE1 = 12
NOISE2 = 15
SELEM  = square(3)

# -----------------------------
# Generar dominios
# -----------------------------
mask1, r1_grid, theta_grid = noisy_circle_mask(SIZE, R1, NOISE1, seed=1)
mask2, r2_grid, _          = noisy_circle_mask(SIZE, R2, NOISE2, seed=2)

# -----------------------------
# Fajas (SIN borde en la lista de animación)
# -----------------------------
thickness, layers, edge_initial = decompose_bands_from_edge(mask1, mask2, SELEM)

# -----------------------------
# Animación (rápida y nítida, sin repintar el borde al final)
# -----------------------------
fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=170)
ax.axis('off')

# mostramos el borde inicial UNA sola vez como contorno tenue
ax.contour(edge_initial.astype(float), levels=[0.5], colors='white', linewidths=1.2)

# imagen a colorear
im = ax.imshow(np.zeros_like(mask1), cmap='viridis',
               vmin=0, vmax=len(layers), interpolation='nearest')

# para acelerar: actualizamos incrementalmente (sin recomputar todo)
canvas = np.zeros_like(mask1, dtype=np.int32)

def init():
    im.set_data(canvas)
    return [im]

def update(frame):
    # activamos SOLO la capa del frame actual (incremental)
    L = layers[frame]
    canvas[L] = frame + 1
    im.set_data(canvas)
    return [im]

ani = FuncAnimation(fig, update, init_func=init,
                    frames=len(layers), interval=250, blit=True)

# Guardá con ffmpeg (rápido); si no tenés ffmpeg, cambiá por PillowWriter(fps=...)
ani.save("fajas_animacion.gif", writer="ffmpeg", fps=6)
plt.close(fig)
print("GIF guardado: fajas_animacion.gif")
