# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 18:07:10 2025

@author: Bautista
"""

# -*- coding: utf-8 -*-
"""
Rugosidad (ξ) por S(q̃) y B(r) — versión Spyder con rango por fracción, por décadas o absoluto.
- Lee NPZ de serie: requiere 'z_curve_al', 'frames', 'files'.
- u(z) = r(z) - <r>, con centroide por frame (A) o fijo del frame 0 (B).
- B(r): permite r_win=(frac_min, frac_max), r_abs=(rmin, rmax) o autodetección.
- S(q̃): permite q_win=(frac_min, frac_max) o autodetección.
- Dibuja pendiente local para ayudarte a fijar rangos.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ============ utilidades numéricas ============

def robust_log(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    return np.log10(x[m]), np.log10(y[m])

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

def sliding_slope(xl, yl, W=21):
    """Pendiente local en log–log con ventana deslizante de W puntos (impar)."""
    xl = np.asarray(xl); yl = np.asarray(yl)
    W = int(W) if int(W)%2==1 else int(W)+1
    half = W//2
    slopes = np.full_like(xl, np.nan, dtype=float)
    for i in range(half, len(xl)-half):
        a, _, _, _ = linear_fit_with_se(xl[i-half:i+half+1], yl[i-half:i+half+1])
        slopes[i] = a
    return slopes

def autodetect_window(xl, yl, min_points=12, win_frac=0.35):
    """Tramo largo con alto R² (verificalo con la curva de pendiente local)."""
    n = len(xl)
    if n < max(min_points, 5):
        return 0, n, 0.0, 0.0, 0.0
    W = max(min_points, int(round(win_frac*n)))
    best = (-np.inf, 0, n, 0.0, 0.0, 0.0)
    for i0 in range(0, max(1, n - W + 1)):
        i1 = i0 + W
        a, b, r2, _ = linear_fit_with_se(xl[i0:i1], yl[i0:i1])
        score = r2 * (i1 - i0)
        if score > best[0]:
            best = (score, i0, i1, a, b, r2)
    _, i0, i1, a, b, r2 = best
    return i0, i1, a, b, r2

# ============ helpers de rango ============

def r_win_from_decades(M, decade_min=0, decade_max=2):
    """
    Convierte 10^a..10^b a fracciones para r_win (sobre 1..M/2).
    Devuelve (rmin_frac, rmax_frac).
    """
    rmin = max(1, int(round(10**decade_min)))
    rmax = int(round(10**decade_max))
    rmax_eff = min(rmax, (M//2) - 1)
    if rmax_eff <= rmin:
        rmax_eff = rmin + 1
    return rmin / (M/2), rmax_eff / (M/2)

# ============ construcción de u(z) ============

def build_u_from_z(z, center_fixed=None):
    z = np.asarray(z, dtype=complex)
    if center_fixed is None:
        cx, cy = np.real(z).mean(), np.imag(z).mean()
    else:
        cx, cy = center_fixed
    r = np.hypot(np.real(z)-cx, np.imag(z)-cy)
    return r - r.mean()

def center_choice(z_frames, option='A'):
    if option.upper() == 'B':
        z0 = np.asarray(z_frames[0], dtype=complex)
        return (np.real(z0).mean(), np.imag(z0).mean())
    return None

# ============ métodos ξ(B) y ξ(S) ============

def xi_from_structure(u, r_win=None, r_abs=None, out_ax=None):
    """
    B(r): si r_abs o r_win están dados, el AJUSTE se hace sobre TODO ese tramo fijo (sin autodetección).
           r_abs = (rmin_abs, rmax_abs) en lags absolutos (p.ej., (1,100))
           r_win = (frac_min, frac_max) sobre el semirango 1..M/2
    Si ambos son None → usa todo el rango y autodetecta un tramo lineal.
    Devuelve: dict con xi, sigma, slope, r2, xl, yl, slopes, i0, i1, selected_min/max.
    """
    y = np.asarray(u, float); M = len(y)
    lags = np.arange(1, M//2)  # 1..M/2-1

    # --- Selección de rango ---
    selected_min = None; selected_max = None
    if r_abs is not None:
        rmin_abs, rmax_abs = int(r_abs[0]), int(r_abs[1])
        rmin_abs = max(1, rmin_abs)
        rmax_abs = min(M//2 - 1, rmax_abs)
        if rmax_abs <= rmin_abs:
            rmax_abs = rmin_abs + 1
        msel = (lags >= rmin_abs) & (lags <= rmax_abs)
        lags = lags[msel]
        selected_min, selected_max = lags[0], lags[-1]
    elif r_win is not None:
        rmin_frac, rmax_frac = r_win
        lmin = max(1, int(np.floor((rmin_frac or 0.0) * (M/2))))
        lmax = min(M//2 - 1, int(np.ceil((rmax_frac or 1.0) * (M/2))))
        msel = (lags >= lmin) & (lags <= lmax)
        lags = lags[msel]
        selected_min, selected_max = lags[0], lags[-1]

    if len(lags) < 5:
        raise ValueError("Muy pocos lags seleccionados para B(r). Ajustá r_abs/r_win.")

    # --- B(r) en el tramo seleccionado ---
    B = np.empty_like(lags, dtype=float)
    for i, d in enumerate(lags):
        diff = y[:-d] - y[d:]
        B[i] = np.mean(diff*diff)

    # logs y pendiente local (DIAGNÓSTICO)
    xl, yl = robust_log(lags.astype(float), B)
    slopes = sliding_slope(xl, yl, W=max(11, len(xl)//15 or 11))

    # --- Ajuste lineal ---
    if (r_abs is not None) or (r_win is not None):
        i0, i1 = 0, len(xl)  # usa TODO el tramo elegido
    else:
        i0, i1, _, _, _ = autodetect_window(xl, yl)

    a, b, r2, se_a = linear_fit_with_se(xl[i0:i1], yl[i0:i1])
    xi = a/2.0
    sxi = se_a/2.0

    # --- Plot ---
    if out_ax is not None:
        ax = out_ax
        ax.loglog(lags, B, '-', lw=1.0, label='B(r)')
        xx = np.logspace(np.log10(lags[0]), np.log10(lags[-1]), 80)
        yy = 10**(a*np.log10(xx) + b)
        ax.loglog(xx, yy, '--', lw=2.2, label=f'fit slope={a:.3f}')
        if (selected_min is not None) and (selected_max is not None):
            ax.axvspan(selected_min, selected_max, alpha=0.12, color='tab:orange')
        ax.set_xlabel('r (muestras)'); ax.set_ylabel('B(r)')
        ax.set_title(f'B(r): ξ={xi:.3f} ± {sxi:.3f}')
        ax.legend(frameon=False)

    return dict(xi=xi, sigma=sxi, slope=a, r2=r2,
                lags=lags, B=B, xl=xl, yl=yl, slopes=slopes,
                i0=i0, i1=i1, selected_min=selected_min, selected_max=selected_max)

def xi_from_spectrum(u, q_win=None, out_ax=None):
    """
    S(q̃): q_win en fracción de k=1..M/2 (None → todo y autodetección).
    """
    y = np.asarray(u, float); M = len(y)
    U = np.fft.fft(y)
    P = (np.abs(U)**2)/M
    k = np.arange(1, M//2)  # positivos
    q = 2*np.pi*k/float(M)
    qtil = 2*np.sin(q/2.0)
    S = P[1:M//2]

    if q_win is not None:
        qmin_frac, qmax_frac = q_win
        i_min = max(0, int(np.floor((qmin_frac or 0.0) * len(qtil))))
        i_max = min(len(qtil)-1, int(np.ceil((qmax_frac or 1.0) * len(qtil))))
        sel = slice(i_min, i_max+1)
        qtil, S = qtil[sel], S[sel]

    if len(qtil) < 5:
        raise ValueError("Muy pocos puntos seleccionados para S(q̃). Ajustá q_win.")

    xl, yl = robust_log(qtil, S)
    slopes = sliding_slope(xl, yl, W=max(11, len(xl)//15 or 11))
    i0, i1, _, _, _ = autodetect_window(xl, yl)
    a, b, r2, se_a = linear_fit_with_se(xl[i0:i1], yl[i0:i1])
    xi = -(a + 1.0)/2.0; sxi = se_a/2.0

    if out_ax is not None:
        ax = out_ax
        ax.loglog(qtil, S, '-', lw=1.0, label='S(q̃)')
        xx = 10**xl[i0:i1]; yy = 10**(a*xl[i0:i1] + b)
        ax.loglog(xx, yy, '--', lw=2.0, label=f'fit slope={a:.3f}')
        ax.set_xlabel('q̃ = 2 sin(q/2)'); ax.set_ylabel('S(q̃)')
        ax.set_title(f'S(q̃): ξ={xi:.3f} ± {sxi:.3f}')
        ax.legend(frameon=False)

    return dict(xi=xi, sigma=sxi, slope=a, r2=r2, qtil=qtil, S=S, xl=xl, yl=yl, slopes=slopes, i0=i0, i1=i1)

# ============ carga de serie y pipeline ============

def load_series(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    Z = d['z_curve_al'] if 'z_curve_al' in d.files else d['z_curves']
    frames = d['frames'] if 'frames' in d.files else np.arange(len(Z))
    files  = d['files']  if 'files'  in d.files else np.array([f'frame_{i:03d}' for i in range(len(Z))])
    return list(Z), frames, files

def analyze_series(npz_path,
                   center='A',
                   r_win=(0.01, 0.25),     # usado si use_autorange=False y r_abs=None
                   q_win=(0.01, 0.25),     # usado si use_autorange=False
                   r_abs=None,             # rango absoluto B(r), p.ej. (1,100); si se pasa, ignora r_win
                   use_autorange=True,     # True => ignora r_win/r_abs y autodetecta
                   save_dir=None,
                   show_plots=True):
    """
    Devuelve dict con arrays: xi_spec, s_spec, xi_B, s_B y detalle por frame.
    """
    z_list, frames, files = load_series(npz_path)
    center_fixed = center_choice(z_list, center)
    res = dict(frames=np.array(frames), files=np.array(files),
               xi_spec=[], s_spec=[], xi_B=[], s_B=[], detail=[])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'frames'), exist_ok=True)

    for i, z in enumerate(z_list):
        u = build_u_from_z(z, center_fixed=center_fixed)

        # dos paneles de ajuste + dos paneles de pendiente local
        if show_plots or save_dir:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        else:
            axs = (None, None)

        if use_autorange:
            mS = xi_from_spectrum(u, q_win=None, out_ax=axs[0])
            mB = xi_from_structure(u, r_win=None, r_abs=None, out_ax=axs[1])
        else:
            mS = xi_from_spectrum(u, q_win=q_win, out_ax=axs[0])
            mB = xi_from_structure(u, r_win=r_win, r_abs=r_abs, out_ax=axs[1])

        # diagnósticos de pendiente local
        if show_plots or save_dir:
            fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3))
            ax2[0].plot(mS['xl'], mS['slopes'], '-', lw=1)
            ax2[0].axvspan(mS['xl'][mS['i0']], mS['xl'][mS['i1']-1], alpha=0.15, color='tab:blue')
            ax2[0].set_title('Pendiente local — log S vs log q̃')
            ax2[0].set_xlabel('log10(q̃)'); ax2[0].set_ylabel('pendiente')

            ax2[1].plot(mB['xl'], mB['slopes'], '-', lw=1)
            ax2[1].axvspan(mB['xl'][mB['i0']], mB['xl'][mB['i1']-1], alpha=0.15, color='tab:orange')
            ax2[1].set_title('Pendiente local — log B vs log r')
            ax2[1].set_xlabel('log10(r)'); ax2[1].set_ylabel('pendiente')

        if save_dir:
            fig.tight_layout(); fig.savefig(os.path.join(save_dir, 'frames', f'frame_{i:03d}_fits.png'), dpi=220)
            plt.close(fig)
            fig2.tight_layout(); fig2.savefig(os.path.join(save_dir, 'frames', f'frame_{i:03d}_localslope.png'), dpi=220)
            plt.close(fig2)

        res['xi_spec'].append(mS['xi']); res['s_spec'].append(mS['sigma'])
        res['xi_B'].append(mB['xi']);   res['s_B'].append(mB['sigma'])
        res['detail'].append(dict(S=mS, B=mB))

    # arrays
    for k in ['xi_spec','s_spec','xi_B','s_B']:
        res[k] = np.array(res[k], float)

    # overview
    if show_plots or save_dir:
        x = np.arange(len(z_list))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.errorbar(x, res['xi_spec'], yerr=res['s_spec'], fmt='o', ms=4, capsize=3, label='ξ espectro')
        ax.errorbar(x, res['xi_B'],    yerr=res['s_B'],    fmt='s', ms=4, capsize=3, label='ξ B(r)')
        ax.set_xlabel('Frame'); ax.set_ylabel('ξ'); ax.grid(alpha=0.3); ax.legend(frameon=False)
        fig.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'xi_overview.png'), dpi=220); plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(9, 3))
        ax2.plot(x, res['xi_spec'] - res['xi_B'], 'o-', lw=1)
        ax2.axhline(0, color='k', lw=1)
        ax2.set_xlabel('Frame'); ax2.set_ylabel('Δξ (espectro − B)')
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        if save_dir:
            fig2.savefig(os.path.join(save_dir, 'xi_delta.png'), dpi=220); plt.close(fig2)

    return res

# Ruta y opciones
npz_path = r"E:\Documents\Labo 6\LuCam-app\analisis\parametrizacion\out_pablo\series_results.npz"
save_dir = r"E:\Documents\Labo 6\LuCam-app\analisis\parametrizacion\rugosidad_out"

center = 'A'             # centroide por frame
use_autorange = False    # ¡importante!: así respeta r_abs
r_abs = (1, 100)         # 10^0 .. 10^2 en lags absolutos
r_win = None             # ignorado si r_abs no es None
q_win = (0.01, 0.25)     # espectro: 1%..25% del semirango (ajustá si querés)

res = analyze_series(npz_path,
                     center='A',
                     r_win=None,
                     q_win=(0.01, 0.25),
                     r_abs=(1, 100),      # fuerza 10^0..10^2
                     use_autorange=False,
                     save_dir=save_dir,
                     show_plots=True)

# Números a mano
xi_spec = res['xi_spec']; xi_B = res['xi_B']
print(f"ξ (espectro): mean={xi_spec.mean():.4f}  std={xi_spec.std(ddof=1):.4f}")
print(f"ξ (B)       : mean={xi_B.mean():.4f}     std={xi_B.std(ddof=1):.4f}")

