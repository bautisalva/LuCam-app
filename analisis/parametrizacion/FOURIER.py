# Fourier-param-s.py
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt

# Opcional: si tenés skimage y tu enhancer, los usamos para sacar contornos
try:
    from skimage.io import imread
    from skimage.filters import threshold_otsu
    from skimage.measure import find_contours
    from scipy.ndimage import uniform_filter
except Exception:
    imread = None
    find_contours = None

# ====== (1) Remuestreo uniforme por longitud de arco ======
def resample_closed_curve_arclength(P_yx, M):
    """P_yx: (N,2) en (y,x). Devuelve P_u (M,2), s (M,), L (perímetro)."""
    P = np.asarray(P_yx, float)
    d = np.diff(np.vstack([P, P[:1]]), axis=0)
    seg = np.hypot(d[:,0], d[:,1])
    s = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    L = s[-1] + seg[-1]
    s_u = np.linspace(0, L, M, endpoint=False)
    y_u = np.interp(s_u, s, P[:,0])
    x_u = np.interp(s_u, s, P[:,1])
    return np.column_stack([y_u, x_u]), s_u, L

# ====== (2) Descriptores de Fourier complejos sobre s ======
def fourier_descriptors_z(P_yx_u, remove_translation=True, norm_scale=None, norm_rotation=None):
    """
    P_yx_u: (M,2) muestreada UNIFORME en s.
    Devuelve dict con: Z (FFT de z), z_mean, scale, rot_phase.
    - remove_translation: quita la media (invar. traslación).
    - norm_scale: si 'l2' normaliza energía; si 'k1' normaliza |Z[1]|.
    - norm_rotation: si 'k1' alinea fase de Z[1] a 0 (invar. rotación).
    """
    y = P_yx_u[:,0]; x = P_yx_u[:,1]
    z = x + 1j*y
    z_mean = z.mean()
    if remove_translation:
        z = z - z_mean

    Z = np.fft.fft(z)  # k=0..M-1
    M = len(Z)

    scale = 1.0
    if norm_scale == 'l2':
        e = np.sqrt(np.sum(np.abs(Z[1:])**2)) + 1e-12
        Z = Z / e; scale = 1.0/e
    elif norm_scale == 'k1':
        a1 = np.abs(Z[1]) + 1e-12
        Z = Z / a1; scale = 1.0/a1

    rot_phase = 0.0
    if norm_rotation == 'k1':
        rot_phase = -np.angle(Z[1])
        rot = np.exp(1j*rot_phase)
        # rotar todos los armónicos excepto DC (equivale a rotación en el plano)
        Z = Z * rot

    return dict(Z=Z, z_mean=z_mean if remove_translation else 0.0,
                scale=scale, rot_phase=rot_phase)

# ====== (3) Reconstrucción trunca con M modos ======
def reconstruct_from_Z(Z, Mkeep, z_mean=0.0):
    """
    Z: FFT completa (M muestras). Mkeep: número de armónicos positivos a retener.
    Devuelve z_rec (M,), en el dominio de la señal muestreada uniforme en s.
    """
    Mtot = len(Z)
    Zt = np.zeros_like(Z)
    Zt[0] = Z[0]
    # armónicos positivos 1..Mkeep y negativos simétricos
    for k in range(1, min(Mkeep, Mtot//2)+1):
        Zt[k] = Z[k]
        Zt[-k] = Z[-k]
    z_rec = np.fft.ifft(Zt) + z_mean
    return z_rec

# ====== (4) Métricas ======
def rel_mse(z_true, z_rec):
    num = np.mean(np.abs(z_true - z_rec)**2)
    den = np.mean(np.abs(z_true - z_true.mean())**2) + 1e-12
    return float(num/den)

# ====== (5) Helper: sacar contorno principal de una imagen ======
def longest_contour_yx(binary_or_image):
    """
    Si recibe imagen grayscale, binariza con Otsu+suavizado; si recibe binaria, usa directo.
    Requiere skimage instalado. Devuelve contorno (y,x) más largo o None.
    """
    if find_contours is None:
        raise RuntimeError("Necesitás scikit-image para esta función.")
    from skimage.filters import threshold_otsu
    from scipy.ndimage import uniform_filter
    img = binary_or_image
    if img.dtype != np.uint8 and img.dtype != bool:
        sm = uniform_filter(img.astype(np.float32), size=3)
        thr = threshold_otsu(sm)
        b = (sm > thr).astype(np.uint8)
    else:
        b = (img > 0).astype(np.uint8)
    cs = find_contours(b.astype(float), level=0.5)
    if not cs:
        return None
    return max(cs, key=lambda c: c.shape[0])

# ====== (6) Pipeline: de contorno -> param s -> FFT -> recon & plots ======
def param_fourier_over_s(contour_yx, Msamples=4096,
                         invariants=('translate',),  # opciones: 'translate','scale_l2','scale_k1','rot_k1'
                         Mkeep_list=(32, 64, 128, 256),
                         do_plots=True, title_tag=""):
    """
    contour_yx: (N,2) en (y,x). Msamples: remuestreo uniforme en s.
    invariants: qué invariantes aplicar (traslación siempre usada si está en lista).
    """
    # 1) Remuestreo por arco
    P_u, s_u, L = resample_closed_curve_arclength(contour_yx, Msamples)
    z_u = P_u[:,1] + 1j*P_u[:,0]

    # 2) Descriptores
    remove_translation = 'translate' in invariants
    norm_scale = ('scale_k1' if 'scale_k1' in invariants else
                  'l2' if 'scale_l2' in invariants else None)
    norm_rotation = 'k1' if 'rot_k1' in invariants else None
    fd = fourier_descriptors_z(P_u, remove_translation=remove_translation,
                               norm_scale=norm_scale, norm_rotation=norm_rotation)
    Z = fd['Z']; z_mean = fd['z_mean']

    # 3) Reconstrucciones y errores
    recons = {}
    errs = {}
    for Mk in Mkeep_list:
        zr = reconstruct_from_Z(Z, Mk, z_mean=z_mean)
        recons[Mk] = zr
        errs[Mk] = rel_mse(z_u, zr)

    if do_plots:
        # (a) Curva original vs recon
        plt.figure(figsize=(5.8,5.8))
        plt.plot(np.real(z_u), np.imag(z_u), 'k-', lw=0.8, label='Original (s-uniforme)')
        colors = plt.cm.viridis(np.linspace(0,1,len(Mkeep_list)))
        for Mk, col in zip(Mkeep_list, colors):
            zr = recons[Mk]
            plt.plot(np.real(zr), np.imag(zr), '-', color=col, lw=1.2, label=f'{Mk} modos, rMSE={errs[Mk]:.3e}')
        plt.axis('equal'); plt.title(f'Fourier sobre s {title_tag}')
        plt.legend(fontsize=8); plt.tight_layout(); plt.show()

        # (b) x(s) e y(s)
        plt.figure(figsize=(8.8,3.2))
        plt.subplot(1,2,1)
        plt.plot(s_u, np.real(z_u), 'k-', lw=0.7, alpha=0.6, label='x(s) orig')
        for Mk, col in zip(Mkeep_list, colors):
            zr = recons[Mk]; plt.plot(s_u, np.real(zr), '-', color=col, lw=1.0, label=f'x_M={Mk}')
        plt.xlabel('s [px]'); plt.ylabel('x(s)'); plt.grid(True, alpha=0.3); plt.legend(fontsize=7)
        plt.subplot(1,2,2)
        plt.plot(s_u, np.imag(z_u), 'k-', lw=0.7, alpha=0.6, label='y(s) orig')
        for Mk, col in zip(Mkeep_list, colors):
            zr = recons[Mk]; plt.plot(s_u, np.imag(zr), '-', color=col, lw=1.0, label=f'y_M={Mk}')
        plt.xlabel('s [px]'); plt.ylabel('y(s)'); plt.grid(True, alpha=0.3); plt.legend(fontsize=7)
        plt.tight_layout(); plt.show()

        # (c) r(s) respecto de centro Kasa o centroide (acá usamos centroide por simplicidad)
        xc, yc = np.real(z_u).mean(), np.imag(z_u).mean()
        r = np.hypot(np.real(z_u)-xc, np.imag(z_u)-yc)
        plt.figure(figsize=(6.6,3.0))
        plt.plot(s_u, r, 'k-', lw=0.8, alpha=0.7, label='r(s) original')
        for Mk, col in zip(Mkeep_list, colors):
            zr = recons[Mk]; rM = np.hypot(np.real(zr)-xc, np.imag(zr)-yc)
            plt.plot(s_u, rM, '-', color=col, lw=1.0, label=f'r_M={Mk}')
        plt.xlabel('s [px]'); plt.ylabel('r(s)'); plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
        plt.title('Perfil radial vs s (centroide)')
        plt.tight_layout(); plt.show()

        # (d) Espectro
        M = len(Z); k = np.arange(M)
        power = (np.abs(Z)/M)**2
        plt.figure(figsize=(6.2,3.2))
        plt.loglog(k[1:M//2], power[1:M//2], '-')
        plt.xlabel('modo k'); plt.ylabel('|Z[k]|^2'); plt.title('Espectro (detalle vs k)')
        plt.grid(True, which='both', alpha=0.3); plt.tight_layout(); plt.show()

    return dict(Z=Z, s=s_u, L=L, z=z_u, recons=recons, errs=errs, z_mean=z_mean)

# ====== (7) Opcional: enganchar con tu FI (usa tu clase si existe) ======
def try_import_enhancer():
    try:
        from Analisis_poco_contorno import ImageEnhancer
    except Exception:
        ImageEnhancer = None
    return ImageEnhancer

def contours_from_image(im):
    """
    Devuelve lista de contornos (y,x). Usa tu ImageEnhancer si existe;
    si no, Otsu + find_contours.
    """
    IE = try_import_enhancer()
    if IE is not None:
        enh = IE(im)
        binary, contornos, _ = enh.procesar(mostrar=False, suavizado=3,
                                            percentil_contornos=99.9, min_dist_picos=8000,
                                            metodo_contorno='binarizacion')
        return contornos
    # Fallback simple
    sm = uniform_filter(im.astype(np.float32), size=3)
    thr = threshold_otsu(sm)
    b = (sm > thr).astype(np.uint8)
    return find_contours(b.astype(float), level=0.5)

def longest_contour(contours):
    if not contours: return None
    return max(contours, key=lambda c: c.shape[0])

# ====== (8) DEMO mínima ======
if __name__ == "__main__":
    # --- Opción A: cargar una imagen de tu FI25 y sacar su contorno principal ---
    # Cambiá "FOLDER" por tu carpeta de FI25 y pattern
    FOLDER = r"C:\ruta\a\tu\FI25"            # <-- poné tu carpeta
    PATTERN = "resta_*.tif"                  # mismo patrón que usás
    USE_FIRST = True

    if imread is not None and find_contours is not None and os.path.isdir(FOLDER):
        files = sorted(glob.glob(os.path.join(FOLDER, PATTERN)))
        if not files:
            print("[INFO] No encontré imágenes; corro demo sintética.")
            USE_FIRST = False
        else:
            im = imread(files[0])
            conts = contours_from_image(im)
            C = longest_contour(conts)
            if C is None or len(C) < 8:
                print("[WARN] No pude extraer contorno; corro demo sintética.")
                USE_FIRST = False
            else:
                print(f"[OK] Contorno length={len(C)}")
                _ = param_fourier_over_s(C, Msamples=4096,
                                         invariants=('translate',),  # podés sumar 'scale_k1','rot_k1'
                                         Mkeep_list=(32,64,128,256),
                                         do_plots=True,
                                         title_tag="(FI25)")
    if not USE_FIRST:
        # --- Opción B: demo sintética NO-UNIVALUADA (r(θ) no sirve) ---
        # hacemos una curva cerrada con “puntas” y reentrancias
        th = np.linspace(0, 2*np.pi, 3000, endpoint=False)
        r = 120 + 6*np.cos(3*th) + 3*np.sin(7*th)
        # pliegues tangenciales para romper univalencia radial
        x = r*np.cos(th) + 5*np.sin(12*th)
        y = r*np.sin(th) + 5*np.cos(9*th)
        P_yx = np.column_stack([y, x])

        _ = param_fourier_over_s(P_yx, Msamples=4096,
                                 invariants=('translate',),      # añadir 'rot_k1','scale_k1' si querés invariantes
                                 Mkeep_list=(32,64,128,256,512),
                                 do_plots=True,
                                 title_tag="(demo sintética)")


#%%

# === FI25: Fourier sobre s para "250 x 10" (usa tu crop y start) ===
import os, re, glob
import numpy as np
import matplotlib.pyplot as plt

# ---------- CONFIG (tomado de tu código) ----------
ROOT_FI = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\Analisis para informe\23-06-2025"  # <- ajustá si hace falta
SUBFOLDER = "250 x 10"                 # CAMPOS_FI_RAW[80]["folder"]
FI_PATTERN = r"resta_*.tif"            # patrón de nombres
FI_REGEX   = r"resta_(\d{8}_\d{6})\.tif"  # orden temporal por timestamp
START, END = 15, None                  # CAMPOS_FI_RAW[80]["start"], ["end"]
CROP = (slice(140, 280), slice(345, 495))  # CAMPOS_FI_RAW[80]["crop"]

# ---------- (1) Helpers de carga y orden ----------
def natural_sort_key(filename: str, regex_key: str):
    m = re.search(regex_key, os.path.basename(filename))
    if m:
        g = m.group(1)
        if g.isdigit(): return (0, int(g))
        if re.fullmatch(r"\d{8}_\d{6}", g): return (1, g)
        return (2, g)
    m2 = re.search(r"(\d+)(?=\.\w+$)", os.path.basename(filename))
    if m2: return (3, int(m2.group(1)))
    return (4, os.path.basename(filename))

def load_stack(folder, pattern_glob, regex_key, crop, start, end):
    from skimage.io import imread
    files = glob.glob(os.path.join(folder, pattern_glob))
    files.sort(key=lambda f: natural_sort_key(f, regex_key))
    files = files[start: end if end is not None else None]
    imgs = []
    for f in files:
        try:
            im = imread(f)
            if crop is not None: im = im[crop]
            imgs.append(im)
        except Exception as e:
            print(f"[WARN] No se pudo cargar {f}: {e}")
    return imgs

# ---------- (2) Binarización/contorno (usa tu enhancer si está) ----------
def fi_get_binary_and_contours(im):
    from skimage.measure import find_contours
    from skimage.filters import threshold_otsu
    from scipy.ndimage import uniform_filter
    # Intentar tu clase:
    try:
        from Analisis_poco_contorno import ImageEnhancer
        enh = ImageEnhancer(im)
        binary, contornos, _ = enh.procesar(
            mostrar=False, suavizado=3,
            percentil_contornos=99.9, min_dist_picos=8000,
            metodo_contorno="binarizacion"
        )
        return (binary > 0).astype(np.uint8), contornos
    except Exception:
        pass
    # Fallback Otsu:
    sm = uniform_filter(im.astype(np.float32), size=3)
    thr = threshold_otsu(sm)
    b = (sm > thr).astype(np.uint8)
    cont = find_contours(b.astype(float), level=0.5)
    return b, cont

def longest_contour(contours):
    if not contours: return None
    return max(contours, key=lambda c: c.shape[0])

# ---------- (3) Remuestreo por longitud de arco ----------
def resample_closed_curve_arclength(P_yx, M):
    P = np.asarray(P_yx, float)
    d = np.diff(np.vstack([P, P[:1]]), axis=0)
    seg = np.hypot(d[:,0], d[:,1])
    s = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    L = s[-1] + seg[-1]
    s_u = np.linspace(0, L, M, endpoint=False)
    y_u = np.interp(s_u, s, P[:,0])
    x_u = np.interp(s_u, s, P[:,1])
    return np.column_stack([y_u, x_u]), s_u, L

# ---------- (4) Fourier sobre s ----------
def fourier_descriptors_z(P_yx_u, remove_translation=True, norm_scale=None, norm_rotation=None):
    y = P_yx_u[:,0]; x = P_yx_u[:,1]; z = x + 1j*y
    z_mean = z.mean() if remove_translation else 0.0
    if remove_translation: z = z - z_mean
    Z = np.fft.fft(z)
    scale = 1.0
    if norm_scale == "l2":
        e = np.sqrt(np.sum(np.abs(Z[1:])**2)) + 1e-12; Z /= e; scale = 1.0/e
    elif norm_scale == "k1":
        a1 = np.abs(Z[1]) + 1e-12; Z /= a1; scale = 1.0/a1
    rot_phase = 0.0
    if norm_rotation == "k1":
        rot_phase = -np.angle(Z[1]); rot = np.exp(1j*rot_phase); Z *= rot
    return dict(Z=Z, z_mean=z_mean, scale=scale, rot_phase=rot_phase)

def reconstruct_from_Z(Z, Mkeep, z_mean=0.0):
    Mtot = len(Z); Zt = np.zeros_like(Z); Zt[0] = Z[0]
    for k in range(1, min(Mkeep, Mtot//2)+1):
        Zt[k] = Z[k]; Zt[-k] = Z[-k]
    return np.fft.ifft(Zt) + z_mean

# ---------- (5) MAIN ----------
if __name__ == "__main__":
    folder = os.path.join(ROOT_FI, SUBFOLDER)
    imgs = load_stack(folder, FI_PATTERN, FI_REGEX, CROP, START, END)
    if len(imgs) == 0:
        raise SystemExit("[ERROR] No se encontraron imágenes. Verificá ROOT_FI/SUBFOLDER/patrón.")

    im = imgs[-1]  # último frame como “C10”
    b, conts = fi_get_binary_and_contours(im)
    C = longest_contour(conts)
    if C is None or len(C) < 8:
        raise SystemExit("[ERROR] No se pudo extraer un contorno suficientemente largo.")

    # Parametrización por s
    Msamples = 4096
    P_u, s_u, L = resample_closed_curve_arclength(C, Msamples)
    z_u = P_u[:,1] + 1j*P_u[:,0]

    # Descriptores + reconstrucciones
    fd = fourier_descriptors_z(P_u, remove_translation=True, norm_scale=None, norm_rotation=None)
    Z, z_mean = fd["Z"], fd["z_mean"]
    Mkeep_list = (64, 128, 256, 512)
    recons = {M: reconstruct_from_Z(Z, M, z_mean=z_mean) for M in Mkeep_list}

    # ---- Plots ----
    # A) Contorno coloreado por s/L
    c = s_u / L
    plt.figure(figsize=(5.2,5.2))
    plt.scatter(P_u[:,1], P_u[:,0], c=c, s=2)
    plt.gca().invert_yaxis(); plt.axis('equal'); plt.title('C10: parametrización por s'); plt.colorbar(label='s/L')
    plt.tight_layout(); plt.show()

    # B) Reconstrucciones
    plt.figure(figsize=(5.6,5.6))
    plt.plot(np.real(z_u), np.imag(z_u), 'k-', lw=0.8, label='Original')
    cols = plt.cm.viridis(np.linspace(0,1,len(Mkeep_list)))
    for (M,col) in zip(Mkeep_list, cols):
        zr = recons[M]
        plt.plot(np.real(zr), np.imag(zr), '-', color=col, lw=1.1, label=f'{M} modos')
    plt.axis('equal'); plt.legend(fontsize=8); plt.title('Fourier sobre s (C10)')
    plt.tight_layout(); plt.show()

    # C) x(s) e y(s)
    plt.figure(figsize=(9,3.2))
    plt.subplot(1,2,1); plt.plot(s_u, np.real(z_u), 'k-', lw=0.8); plt.xlabel('s [px]'); plt.ylabel('x(s)'); plt.grid(True, alpha=0.3)
    plt.subplot(1,2,2); plt.plot(s_u, np.imag(z_u), 'k-', lw=0.8); plt.xlabel('s [px]'); plt.ylabel('y(s)'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # D) r(s) respecto de centroide
    xc, yc = np.real(z_u).mean(), np.imag(z_u).mean()
    r = np.hypot(np.real(z_u)-xc, np.imag(z_u)-yc)
    plt.figure(figsize=(6.4,3.0)); plt.plot(s_u, r, 'k-', lw=0.8)
    plt.xlabel('s [px]'); plt.ylabel('r(s) [px]'); plt.grid(True, alpha=0.3); plt.title('Perfil radial vs s (centroide)')
    plt.tight_layout(); plt.show()

    # Guardar CSV (opcional)
    # np.savetxt("C10_param_250x10.csv", np.column_stack([s_u, P_u[:,1], P_u[:,0]]), delimiter=",", header="s,x,y", comments="")
