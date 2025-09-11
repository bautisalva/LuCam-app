# -*- coding: utf-8 -*-
"""
alpha_from_series_fixedK_options.py
===================================

Calcula el exponente de rugosidad α (Método 5, espectro) en cada frame de
`series_results.npz`, usando **K fijo** para todas las curvas. Incluye dos variantes:

- **Opción A (default)**: centroide por frame + ventana automática de ajuste en el espectro.
- **Opción B**: centroide fijo (del primer frame) + ventana automática de ajuste.

La ventana automática se elige en log–log maximizando R² en tramos de tamaño relativo.

Uso:
----
python alpha_from_series_fixedK_options.py \
  --series_npz "C:\\...\\series_results.npz" \
  --out_dir "alpha_out" \
  --K_fixed 4096 \
  --option A

"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- utils ----------
def ensure_1d_float(a):
    return np.array(a, dtype=float).reshape(-1)

def robust_log(x, y):
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    return np.log10(x[m]), np.log10(y[m])

def linear_fit(x, y):
    A = np.column_stack((x, np.ones_like(x)))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    yhat = a*x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-30
    r2 = 1.0 - ss_res/ss_tot
    return float(a), float(b), float(r2)

def autodetect_window(xl, yl, min_points=10, win_frac=0.3):
    n = len(xl)
    W = max(min_points, int(round(win_frac*n)))
    best = (-np.inf, 0, 0, 0, 0, 0)
    for i0 in range(0, n - W + 1):
        i1 = i0 + W
        a, b, r2 = linear_fit(xl[i0:i1], yl[i0:i1])
        score = r2 * (i1 - i0)
        if score > best[0]:
            best = (score, i0, i1, a, b, r2)
    _, i0, i1, a, b, r2 = best
    return i0, i1, a, b, r2

# ---------- espectro ----------
def estimate_alpha_auto(h, out_png):
    y = np.asarray(h, float)
    M = len(y)
    Y = np.fft.fft(y)
    S = (np.abs(Y)**2) / M
    k = np.abs(np.fft.fftfreq(M, d=1.0))
    mask = (k > 0)
    k, S = k[mask], S[mask]
    xl, yl = robust_log(k, S)
    i0, i1, a, b, r2 = autodetect_window(xl, yl)
    alpha = -(a + 1.0)/2.0
    # error estándar aprox. a partir de residuales
    seg_x, seg_y = xl[i0:i1], yl[i0:i1]
    a2, b2, r2 = linear_fit(seg_x, seg_y)
    resid = seg_y - (a2*seg_x + b2)
    sigma_a = np.std(resid)/np.sqrt(len(seg_x))/np.std(seg_x)
    sigma_alpha = sigma_a/2.0
    # plot
    plt.figure(figsize=(6.2, 4.2))
    plt.loglog(k, S, '-', lw=1.1, label='S(k)')
    xx = 10**seg_x
    yy = 10**(a2*seg_x + b2)
    plt.loglog(xx, yy, '--', lw=2.0, label=f"slope={a2:.3f}")
    plt.xlabel('k')
    plt.ylabel('S(k)')
    plt.title(f'α={alpha:.3f} ± {sigma_alpha:.3f}')
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()
    return dict(alpha=alpha, sigma_alpha=sigma_alpha, slope=a2, r2=r2)

# ---------- pipeline ----------
def run(series_npz, out_dir, K_fixed, option):
    os.makedirs(out_dir, exist_ok=True)
    d = np.load(series_npz, allow_pickle=True)
    frames = d['frames']
    files  = d['files']
    z_curves = d['z_curve_al']

    # centroide fijo si opción B
    if option == 'B':
        z0 = np.array(z_curves[0], dtype=complex)
        cx0, cy0 = np.real(z0).mean(), np.imag(z0).mean()

    alphas = []; sigmas = []; metas = []
    frames_dir = os.path.join(out_dir, 'frames'); os.makedirs(frames_dir, exist_ok=True)

    for i in range(len(frames)):
        z = np.array(z_curves[i], dtype=complex)
        if option == 'A':
            cx, cy = np.real(z).mean(), np.imag(z).mean()
        else:
            cx, cy = cx0, cy0
        r = np.hypot(np.real(z)-cx, np.imag(z)-cy)
        h = r - r.mean()
        out_png = os.path.join(frames_dir, f'spectrum_{i:03d}.png')
        meta = estimate_alpha_auto(h, out_png)
        alphas.append(meta['alpha']); sigmas.append(meta['sigma_alpha']); metas.append(meta)

    alphas = np.array(alphas, float)
    sigmas = np.array(sigmas, float)

    # CSV
    with open(os.path.join(out_dir, 'alpha_per_frame.csv'), 'w', encoding='utf-8') as f:
        f.write('frame,file,alpha,sigma_alpha,slope,R2\n')
        for i, m in enumerate(metas):
            f.write(f"{int(frames[i])},\"{str(files[i])}\",{m['alpha']:.6f},{m['sigma_alpha']:.6f},{m['slope']:.6f},{m['r2']:.5f}\n")

    # overview
    x = np.arange(len(alphas))
    plt.figure(figsize=(8.0, 4.2))
    plt.errorbar(x, alphas, yerr=sigmas, fmt='o', lw=1.0, ms=4, capsize=3)
    plt.xlabel('Frame')
    plt.ylabel('α (espectro)')
    plt.title(f'α por frame — opción {option}')
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'alpha_overview.png'), dpi=220); plt.close()

    # resumen
    np.savez(os.path.join(out_dir, 'alpha_results.npz'),
             frames=frames, files=files, alpha=alphas, sigma_alpha=sigmas,
             metas=np.array(metas, dtype=object), option=option)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Alpha (Método 5, espectro) — opción {option}\n')
        f.write(f"Promedio: {alphas.mean():.5f}  |  Desvío: {alphas.std(ddof=1):.5f}\n")

    print('[OK] α calculado con opción', option, 'Resultados en', out_dir)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description='Alpha (espectro) con K fijo por frame, opción A o B')
    p.add_argument('--series_npz', type=str, default=r'C:\Users\Marina\Documents\Labo 6\LuCam-app\analisis\parametrizacion\out_pablo\series_results.npz' )
    p.add_argument('--out_dir', type=str, default='alpha_out')
    p.add_argument('--K_fixed', type=int, default=4096)
    p.add_argument('--option', type=str, default='B', choices=['A','B'], help='A=centroide por frame, B=centroide fijo del primer frame')
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.isfile(args.series_npz):
        raise FileNotFoundError(args.series_npz)
    run(args.series_npz, args.out_dir, args.K_fixed, args.option)

if __name__ == '__main__':
    main()
