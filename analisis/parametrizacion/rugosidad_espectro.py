# -*- coding: utf-8 -*-
"""
rugosidad_tesis_compare.py
==========================

Compara el **exponente de rugosidad ξ (≡ α)** usando **exactamente** las definiciones del
capítulo que pasaste:

- **Función rugosidad**:  B(r) = ⟨[u(z+r) − u(z)]²⟩ ~ r^{2ξ}  ⇒  ξ = (pendiente)/2.
- **Factor de estructura**: S(q) = û(q) û(−q) ~ q^{−(1+2ξ)}  ⇒  ξ = (−pendiente − 1)/2,
  usando la **coordenada discreta**  q̃ = 2·sin(q/2)  con  q = 2πk/M  (como en la tesis).

Donde el perfil univaluado es  u(z) = r(z) − ⟨r⟩  con  r(z) = |z(z) − c|.

Opciones clave (coherencia y transparencia):
- **Centroide**:  A = centroide por frame  |  B = centroide fijo (del frame 0).
- **Ventanas de ajuste**: automáticas por defecto, con límites manuales opcionales
  para r y q̃ si querés controlar el rango (sin “sobreajustar” rectas).
- **K fijo aguas arriba**: este script NO re-sintetiza geometría; asume que el NPZ
  ya tiene z_curve_al remuestreado con M uniforme y (idealmente) K fijo.

Salidas:
- Por frame:  *_spectrum.png  y  *_structure.png  con tramo usado y pendiente.
- CSV:       alpha_compare_per_frame.csv  (ξ por ambos métodos + R² y tramos).
- Figuras:   alpha_compare_overview.png  y  alpha_compare_delta.png.

Uso:
-----
python rugosidad_tesis_compare.py \
  --series_npz "C:\\...\\series_results.npz" \
  --out_dir "alpha_tesis_out" \
  --option A \
  --rmin_frac 0.001 --rmax_frac 0.25 \
  --qmin_frac 0.001 --qmax_frac 0.25

Notas:
- *_frac* se refieren a fracciones de los rangos discretos: r∈[1..M/2], q=k→q̃ con k∈[1..M/2].
- Si omitís límites, el detector automático busca el mejor tramo lineal largo y con alto R².
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Utils ----------------
def ensure_1d_float(a):
    return np.array(a, dtype=float).reshape(-1)

def center_choice(z_frames, option):
    if option == 'B':
        z0 = np.asarray(z_frames[0], dtype=complex)
        return (np.real(z0).mean(), np.imag(z0).mean())
    return None  # A: por frame

def build_u_from_z(z, center_fixed=None):
    zc = np.asarray(z, dtype=complex)
    if center_fixed is None:
        cx, cy = np.real(zc).mean(), np.imag(zc).mean()
    else:
        cx, cy = center_fixed
    r = np.hypot(np.real(zc) - cx, np.imag(zc) - cy)
    return r - r.mean()

def linear_fit_with_se(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.column_stack((x, np.ones_like(x)))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    yhat = a*x + b
    resid = y - yhat
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-30
    r2 = 1.0 - ss_res/ss_tot
    n  = max(len(x) - 2, 1)
    s2 = ss_res / n
    Sxx = np.sum((x - x.mean())**2) + 1e-30
    se_a = np.sqrt(s2 / Sxx)
    return float(a), float(b), float(r2), float(se_a)

def robust_log(x, y):
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    return np.log10(x[m]), np.log10(y[m])

def autodetect_window(xl, yl, min_points=10, win_frac=0.35):
    n = len(xl)
    W = max(min_points, int(round(win_frac*n)))
    best = (-np.inf, 0, 0, 0.0, 0.0, 0.0)
    for i0 in range(0, n - W + 1):
        i1 = i0 + W
        a, b, r2, _ = linear_fit_with_se(xl[i0:i1], yl[i0:i1])
        score = r2 * (i1 - i0)
        if score > best[0]:
            best = (score, i0, i1, a, b, r2)
    _, i0, i1, a, b, r2 = best
    return i0, i1, a, b, r2

# ---------------- Métodos según tesis ----------------

def xi_from_structure(u, out_png, rmin_frac=None, rmax_frac=None):
    y = np.asarray(u, float); M = len(y)
    lags = np.arange(1, M//2)
    # límites manuales (en fracción de M/2)
    if rmin_frac is not None:
        lmin = max(1, int(np.floor(rmin_frac * (M/2))))
    else:
        lmin = 1
    if rmax_frac is not None:
        lmax = min(M//2 - 1, int(np.ceil(rmax_frac * (M/2))))
    else:
        lmax = M//2 - 1
    lags = lags[(lags >= lmin) & (lags <= lmax)]

    B = np.empty_like(lags, dtype=float)
    for i, d in enumerate(lags):
        diff = y[:-d] - y[d:]
        B[i] = np.mean(diff*diff)
    r = lags.astype(float)  # unidad de muestra; la pendiente no cambia con reescala lineal

    xl, yl = robust_log(r, B)
    i0, i1, a, b, r2 = autodetect_window(xl, yl)
    a_seg, b_seg, r2_seg, se_a = linear_fit_with_se(xl[i0:i1], yl[i0:i1])
    xi = a_seg / 2.0
    sigma_xi = se_a / 2.0

    # plot
    plt.figure(figsize=(6.2, 4.2))
    plt.loglog(r, B, '-', lw=1.1, label='B(r)')
    xx = 10**xl[i0:i1]; yy = 10**(a_seg*xl[i0:i1] + b_seg)
    plt.loglog(xx, yy, '--', lw=2.0, label=f"fit slope={a_seg:.3f}")
    plt.xlabel('r (muestras)'); plt.ylabel('B(r)')
    plt.title(f'B(r) — ξ={xi:.3f} ± {sigma_xi:.3f} (1σ)')
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

    return dict(xi=xi, sigma_xi=sigma_xi, slope=a_seg, r2=r2_seg, i0=i0, i1=i1)


def xi_from_spectrum(u, out_png, qmin_frac=None, qmax_frac=None):
    y = np.asarray(u, float); M = len(y)
    U = np.fft.fft(y)
    P = (np.abs(U)**2) / M  # potencia discreta
    k = np.arange(M)  # índice discreto
    # usar sólo k=1..M/2 (positivas)
    k = k[1:M//2]
    q = 2*np.pi*k/float(M)
    q_tilde = 2*np.sin(q/2.0)  # coordenada discreta de la tesis
    S = P[1:M//2]

    # límites manuales en fracción de k-range
    if qmin_frac is not None:
        i_min = max(0, int(np.floor(qmin_frac * len(q_tilde))))
    else:
        i_min = 0
    if qmax_frac is not None:
        i_max = min(len(q_tilde)-1, int(np.ceil(qmax_frac * len(q_tilde))))
    else:
        i_max = len(q_tilde)-1
    sel = slice(i_min, i_max+1)
    qv, Sv = q_tilde[sel], S[sel]

    xl, yl = robust_log(qv, Sv)
    i0, i1, a, b, r2 = autodetect_window(xl, yl)
    a_seg, b_seg, r2_seg, se_a = linear_fit_with_se(xl[i0:i1], yl[i0:i1])
    xi = -(a_seg + 1.0)/2.0
    sigma_xi = se_a/2.0

    # plot
    plt.figure(figsize=(6.2, 4.2))
    plt.loglog(qv, Sv, '-', lw=1.1, label='S(q̃)')
    xx = 10**xl[i0:i1]; yy = 10**(a_seg*xl[i0:i1] + b_seg)
    plt.loglog(xx, yy, '--', lw=2.0, label=f"fit slope={a_seg:.3f}")
    plt.xlabel('q̃ = 2 sin(q/2)'); plt.ylabel('S(q̃)')
    plt.title(f'S(q̃) — ξ={xi:.3f} ± {sigma_xi:.3f} (1σ)')
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

    return dict(xi=xi, sigma_xi=sigma_xi, slope=a_seg, r2=r2_seg, i0=i0, i1=i1)

# ---------------- Pipeline ----------------

def run(series_npz, out_dir, option='A', rmin_frac=None, rmax_frac=None, qmin_frac=None, qmax_frac=None):
    os.makedirs(out_dir, exist_ok=True)
    d = np.load(series_npz, allow_pickle=True)
    frames = d['frames']
    files  = d['files']
    z_curves = d['z_curve_al']

    center_fixed = center_choice(z_curves, option)

    frames_dir = os.path.join(out_dir, 'frames'); os.makedirs(frames_dir, exist_ok=True)
    xi_spec = []; s_spec = []; xi_str = []; s_str = []

    for i in range(len(frames)):
        u = build_u_from_z(z_curves[i], center_fixed=center_fixed)
        mS = xi_from_spectrum(u, os.path.join(frames_dir, f'frame_{i:03d}_spectrum.png'),
                              qmin_frac=qmin_frac, qmax_frac=qmax_frac)
        mB = xi_from_structure(u, os.path.join(frames_dir, f'frame_{i:03d}_structure.png'),
                               rmin_frac=rmin_frac, rmax_frac=rmax_frac)
        xi_spec.append(mS['xi']); s_spec.append(mS['sigma_xi'])
        xi_str.append(mB['xi']);  s_str.append(mB['sigma_xi'])

    xi_spec = np.array(xi_spec, float); s_spec = np.array(s_spec, float)
    xi_str  = np.array(xi_str,  float); s_str  = np.array(s_str,  float)

    # CSV
    with open(os.path.join(out_dir, 'alpha_compare_per_frame.csv'), 'w', encoding='utf-8') as f:
        f.write('frame,file,xi_spectrum,sigma_spectrum,xi_structure,sigma_structure\n')
        for i in range(len(frames)):
            f.write(f"{int(frames[i])},\"{str(files[i])}\",{xi_spec[i]:.6f},{s_spec[i]:.6f},{xi_str[i]:.6f},{s_str[i]:.6f}\n")

    # Overview comparativo y delta
    x = np.arange(len(frames))
    plt.figure(figsize=(8.6, 4.4))
    plt.errorbar(x, xi_spec, yerr=s_spec, fmt='o', ms=4, capsize=3, label='ξ espectro')
    plt.errorbar(x, xi_str,  yerr=s_str,  fmt='s', ms=4, capsize=3, label='ξ B(r)')
    plt.xlabel('Frame'); plt.ylabel('ξ')
    plt.title(f'ξ por ambos métodos — opción {option}')
    plt.grid(alpha=0.3); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'alpha_compare_overview.png'), dpi=220); plt.close()

    plt.figure(figsize=(8.6, 3.6))
    diff = xi_spec - xi_str
    plt.plot(x, diff, 'o-', lw=1.0)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel('Frame'); plt.ylabel('Δξ (espectro − B)')
    plt.title('Diferencia de ξ por método')
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'alpha_compare_delta.png'), dpi=220); plt.close()

    # NPZ
    np.savez(os.path.join(out_dir, 'alpha_compare_results.npz'),
             frames=frames, files=files,
             xi_spectrum=xi_spec, sigma_spectrum=s_spec,
             xi_structure=xi_str, sigma_structure=s_str,
             option=option,
             rmin_frac=rmin_frac, rmax_frac=rmax_frac,
             qmin_frac=qmin_frac, qmax_frac=qmax_frac)

    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Comparación de ξ (espectro vs B) — opción {option}\n')
        f.write(f"Promedios  |  ξ_spec={xi_spec.mean():.4f}  ξ_B={xi_str.mean():.4f}  Δ={np.mean(diff):.4f}\n")
        f.write(f"Desvíos    |  σ_spec={xi_spec.std(ddof=1):.4f}  σ_B={xi_str.std(ddof=1):.4f}\n")

    print('[OK] Comparación (tesis) lista. Resultados en', out_dir)

# ---------------- CLI ----------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Rugosidad (ξ) por B(r) y S(q̃) tal como en la tesis.')
    ap.add_argument('--series_npz', default=r"E:\Documents\Labo 6\LuCam-app\analisis\parametrizacion\out_pabloseries_results.npz", type=str)
    ap.add_argument('--out_dir', default='alpha_tesis_out', type=str)
    ap.add_argument('--option', default='A', choices=['A','B'], help='A: centroide por frame, B: centroide del frame 0')
    ap.add_argument('--rmin_frac', type=float, default=None)
    ap.add_argument('--rmax_frac', type=float, default=None)
    ap.add_argument('--qmin_frac', type=float, default=None)
    ap.add_argument('--qmax_frac', type=float, default=None)
    args = ap.parse_args()
    if not os.path.isfile(args.series_npz):
        raise FileNotFoundError(args.series_npz)
    run(args.series_npz, args.out_dir, option=args.option,
        rmin_frac=args.rmin_frac, rmax_frac=args.rmax_frac,
        qmin_frac=args.qmin_frac, qmax_frac=args.qmax_frac)
