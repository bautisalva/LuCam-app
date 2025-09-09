# =============================================================================
# RUGOSIDAD: B(r) y parametrizaciones vs M (Spyder/Jupyter friendly)
#
# - Carga series_results.npz (de tu pipeline).
# - Toma el frame del medio.
# - Barre M en tu lista; para cada M:
#     * re-muestrea r(s) periódicamente a M puntos uniformes en s,
#     * calcula B(r)=<[u(s+r)-u(s)]^2>, u=r-<r> (promedio circular),
#     * ajusta B(r) ~ B0*(r/r0)^(2*xi) en r∈[RMIN,RMAX] px (cumpliendo r ≥ 3·ds),
#       o bien elige automáticamente la subventana más lineal dentro de [RMIN,RMAX].
# - Grafica:
#     * B(r) superpuestas por M,
#     * xi vs M, R2 & Nfit vs M, linealidad por mitades,
#     * parametrizaciones (XY, x(s/L), y(s/L), r(s/L)) para todos los M.
# - Reporta M “óptimo” con criterios prácticos (puntos, R2, linealidad y estabilidad).
#
# Salidas (prefijo OUT_PREFIX):
#   - OUT_PREFIX_Br_multiM.png
#   - OUT_PREFIX_xi_vs_M.png
#   - OUT_PREFIX_quality_vs_M.png
#   - OUT_PREFIX_linearity_vs_M.png
#   - OUT_PREFIX_param_XY.png (si hay z_curve_al)
#   - OUT_PREFIX_param_xys_vs_snorm.png (si hay z_curve_al)
#   - OUT_PREFIX_param_r_from_z_vs_snorm.png (si hay z_curve_al)
#   - OUT_PREFIX_param_r_vs_snorm.png (si NO hay z_curve_al)
#   - OUT_PREFIX_resumen.csv (tabla con M, xi, B0, R2, rmin_fit, rmax_fit)
# =============================================================================

import os, csv
import numpy as np
import matplotlib.pyplot as plt

# ============================== CONFIG =======================================
NPZ_PATH = r"C:\Users\Tomas\Desktop\FACULTAD\LABO 6\LuCam-app\analisis\parametrizacion\out_pablo\series_results.npz"
OUT_PREFIX = "una_foto_Msweep"

# Lista de M (tu lista completa, no solo potencias de 2)
M_CANDIDATES = [2048, 2187, 3125, 4096, 4374, 4913, 6561, 6859, 7776, 8192]

# Rango de ajuste “fijo” de B(r)
RMIN_REQ = 1.0     # px
RMAX_REQ = 15.0    # px

# Seguridad de resolución y mínimos para ajustar
MIN_PTS_FIT = 8     # puntos mínimos en la ventana de ajuste
DS_SAFETY   = 3.0   # exigimos r >= 3*ds
R2_MIN      = 0.98  # umbral “bueno” de ajuste (solo para seleccionar M óptimo)
NMIN_FIT    = 30    # puntos mínimos “cómodos” (solo para seleccionar M óptimo)
LINEARITY_TOL   = 0.05  # |pend(1a mitad) - pend(2a mitad)| (log-log)
STABILITY_TOL   = 0.02  # |xi(M) - xi(M_mayor)| para estabilidad

# Ventana automática (opcional) para encontrar la subventana más lineal dentro de [RMIN_REQ, RMAX_REQ]
USE_AUTO_WINDOW = False      # True → busca la subventana más lineal y la usa para el ajuste
AUTO_MIN_PTS    = 30         # puntos mínimos en la subventana auto
AUTO_MIN_LOG_SPAN = 0.30     # ancho mínimo en décadas (log10 ~ 0.3 → ~ factor 2 en r)

# Qué bloques ejecutar
RUN_SWEEP_B          = True   # calcular B(r) y xi para todos los M
RUN_PLOT_PARAMETROS  = True   # graficar parametrizaciones para todos los M
RUN_SELECT_M_OPTIMO  = False   # seleccionar M “óptimo” con criterios prácticos
# =============================================================================


# ============================== HELPERS ======================================

def periodic_resample_1d(y, M_new):
    """
    Re-muestreo periódico (interp lineal) de y(t) en t∈[0,1).
    No requiere L; asume muestreo uniforme.
    """
    y = np.asarray(y, float)
    M0 = len(y)
    t0 = np.linspace(0.0, 1.0, M0, endpoint=False)
    t1 = np.linspace(0.0, 1.0, M_new, endpoint=False)
    t0_ext = np.concatenate([t0, [1.0]])
    y_ext  = np.concatenate([y,  y[:1]])
    return np.interp(t1, t0_ext, y_ext)

def structure_function_B_of_u(u):
    """
    B[m] = < (u[i+m] - u[i])^2 >_i, promedio circular.
    Implementación FFT: B = 2(Var - C[m]) con C autocovarianza circular.
    """
    u = np.asarray(u, float)
    u0 = u - u.mean()
    U = np.fft.fft(u0)
    R = np.fft.ifft(U * np.conj(U)).real
    N = len(u0)
    C = R / N
    Var = C[0]
    return 2.0 * (Var - C)

def linfit_loglog(r, B):
    """
    Ajuste lineal en (log r, log B).
    Devuelve pendiente m, intercepto b, y R^2.
    """
    x = np.log(r); y = np.log(B)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = m*x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-12
    R2 = 1.0 - ss_res/ss_tot
    return float(m), float(b), float(R2)

def slope_halves_loglog(r, B):
    """
    Pendientes en 1ª y 2ª mitad de la ventana (log-log) para chequear linealidad.
    """
    n = len(r); mid = n // 2
    m1, _, _ = linfit_loglog(r[:mid], B[:mid])
    m2, _, _ = linfit_loglog(r[mid:], B[mid:])
    return m1, m2

def best_loglog_window(r, B, min_pts=30, min_log_span=0.3):
    """
    Busca subventana [i:j] con máximo R^2 en log-log, sujeta a:
      - al menos 'min_pts' puntos,
      - ancho en log(r) >= min_log_span.
    Retorna dict con R2, i, j, m, b, rmin, rmax; o None si no existe.
    """
    x = np.log(r); y = np.log(B)
    n = len(x); best = None
    for i in range(0, n - min_pts):
        for j in range(i + min_pts, n + 1):
            if (x[j-1] - x[i]) < min_log_span:
                continue
            A = np.vstack([x[i:j], np.ones(j-i)]).T
            m, b = np.linalg.lstsq(A, y[i:j], rcond=None)[0]
            yhat = m*x[i:j] + b
            ss_res = np.sum((y[i:j] - yhat)**2)
            ss_tot = np.sum((y[i:j] - y[i:j].mean())**2) + 1e-12
            R2 = 1.0 - ss_res/ss_tot
            cand = (R2, i, j, m, b)
            if (best is None) or (cand[0] > best[0]):
                best = cand
    if best is None:
        return None
    R2, i, j, m, b = best
    return dict(R2=float(R2), i=int(i), j=int(j), m=float(m), b=float(b),
                rmin=float(np.exp(x[i])), rmax=float(np.exp(x[j-1])))

def load_npz_and_pick_middle_frame(npz_path):
    """
    Carga series_results.npz y retorna (frame_id, L, r_base, has_z, x_base, y_base).
    Si no hay z_curve_al, has_z=False y x_base/y_base son None.
    """
    if not os.path.isfile(npz_path):
        raise SystemExit(f"[ERROR] No existe el archivo NPZ: {npz_path}")
    data   = np.load(npz_path, allow_pickle=True)
    frames = data["frames"]
    Ls     = data["L"].astype(float)
    r_al   = data["r_al"]
    mid    = len(frames) // 2
    frame_id = int(frames[mid])
    L        = float(Ls[mid])
    r_base   = np.asarray(r_al[mid], float)
    has_z = ("z_curve_al" in data)
    x_base = y_base = None
    if has_z:
        z_base = np.asarray(data["z_curve_al"][mid], complex)
        x_base = np.real(z_base); y_base = np.imag(z_base)
    print(f"[INFO] Frame medio: {frame_id} | L = {L:.2f} px | z_curve_al={'sí' if has_z else 'no'}")
    return frame_id, L, r_base, has_z, x_base, y_base

# ============================== CORE: B(r) vs M ==============================

def compute_B_and_fit_for_M(L, r_base, M, rmin_req, rmax_req,
                            min_pts_fit=8, ds_safety=3.0,
                            use_auto_window=False, auto_min_pts=30, auto_min_log_span=0.3):
    """
    Para un M dado:
      1) re-muestrea r(s) a M,
      2) calcula B(r) circular,
      3) define rmin/rmax efectivos (≥ 3·ds y ≤ L/2),
      4) ajusta en ventana fija [rmin,rmax] o subventana “óptima” (use_auto_window),
      5) devuelve dict con métricas y datos de plotting.
    """
    ds = L / M
    # re-muestreo periódico
    rM = periodic_resample_1d(r_base, M)
    u  = rM - rM.mean()

    # estructura B(r)
    B = structure_function_B_of_u(u)
    m = np.arange(M, dtype=float)
    r_sep = m * ds

    mask_half = (r_sep <= 0.5*L)
    r_use = r_sep[mask_half]
    B_use = B[mask_half]

    # rango efectivo
    rmin_eff = max(rmin_req, ds_safety*ds)
    rmax_eff = min(rmax_req, 0.5*L - ds)

    valid = (r_use >= rmin_eff) & (r_use <= rmax_eff) & np.isfinite(B_use) & (B_use > 0)
    r_fit = r_use[valid]; B_fit = B_use[valid]
    if r_fit.size < min_pts_fit:
        return dict(M=M, ds=ds, ok=False, reason="few_points",
                    rmin_eff=rmin_eff, rmax_eff=rmax_eff)

    # Ajuste (fijo o auto)
    if use_auto_window:
        win = best_loglog_window(r_fit, B_fit, min_pts=auto_min_pts, min_log_span=auto_min_log_span)
        if win is None:
            return dict(M=M, ds=ds, ok=False, reason="no_auto_window",
                        rmin_eff=rmin_eff, rmax_eff=rmax_eff)
        # usar subventana óptima
        sl = slice(win["i"], win["j"])
        m_glob, b_glob, R2 = float(win["m"]), float(win["b"]), float(win["R2"])
        rmin_used, rmax_used = float(win["rmin"]), float(win["rmax"])
        r_plot, B_plot = r_fit, B_fit  # para dibujar nube completa
        r_line = np.linspace(rmin_used, rmax_used, 200)
    else:
        # ventana fija
        m_glob, b_glob, R2 = linfit_loglog(r_fit, B_fit)
        rmin_used, rmax_used = float(r_fit.min()), float(r_fit.max())
        r_plot, B_plot = r_fit, B_fit
        r_line = np.linspace(rmin_used, rmax_used, 200)

    xi = 0.5 * m_glob
    B0 = float(np.exp(b_glob))

    # “linealidad” por mitades
    m1, m2 = slope_halves_loglog(r_plot, B_plot)
    m_diff = abs(m1 - m2)

    return dict(
        M=M, ds=ds, ok=True,
        xi=float(xi), B0=float(B0), R2=float(R2),
        rmin_used=rmin_used, rmax_used=rmax_used,
        r_plot=r_use, B_plot=B_use,   # nube recortada a L/2 (no solo ventana)
        r_line=r_line, B_line=B0*(r_line**(2.0*xi)),
        m_diff=float(m_diff),
        Nfit=int(r_plot.size)  # puntos en la nube útil (no solo la subventana)
    )

# ============================== PLOTTING =====================================

def plot_B_multiM(frame_id, L, results, out_prefix, rmin_req, rmax_req):
    plt.figure(figsize=(6.8, 5.0))
    used = 0
    for res, col in zip(results, plt.cm.viridis(np.linspace(0,1,len(results)))):
        if not res["ok"]: continue
        used += 1
        plt.loglog(res["r_plot"][1:], res["B_plot"][1:], '.', ms=3, color=col, alpha=0.65,
                   label=f"M={res['M']} | ξ={res['xi']:.3f}")
        plt.loglog(res["r_line"], res["B_line"], '-', lw=1.8, color=col)
    plt.xlabel("r (px de arco)")
    plt.ylabel("B(r)")
    title = f"B(r) — frame {frame_id} (L={L:.1f}) — ajuste en [{rmin_req},{rmax_req}] px"
    plt.title(title)
    plt.grid(True, which='both', alpha=0.3)
    if used > 0:
        plt.legend(fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(f"{out_prefix}_Br_multiM.png", dpi=220); plt.show()

def plot_quality_vs_M(frame_id, L, results, out_prefix, mark_x=None):
    Ms   = np.array([r["M"] for r in results], int)
    xis  = np.array([r["xi"] if r["ok"] else np.nan for r in results], float)
    R2s  = np.array([r["R2"] if r["ok"] else np.nan for r in results], float)
    Nfit = np.array([r["Nfit"] if r["ok"] else 0 for r in results], int)
    mdif = np.array([r["m_diff"] if r["ok"] else np.nan for r in results], float)

    # xi vs M
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(Ms, xis, 'o-', lw=1.2)
    if mark_x is not None:
        plt.axvline(mark_x, color='tab:red', ls='--', lw=1.2, label=f"M óptimo = {mark_x}")
        plt.legend()
    plt.xlabel("M"); plt.ylabel("Exponente de rugosidad ξ")
    plt.title(f"ξ vs M — frame {frame_id} (L={L:.1f}px)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_prefix}_xi_vs_M.png", dpi=220); plt.show()

    # R2 & Nfit
    fig, ax1 = plt.subplots(figsize=(6.6, 4.2))
    ax1.plot(Ms, R2s, 'o-', lw=1.2, color='tab:blue', label="R²")
    ax1.axhline(R2_MIN, color='tab:blue', ls='--', lw=1.0, alpha=0.7)
    ax1.set_ylabel("R²"); ax1.set_ylim(0.0, 1.01)
    ax2 = ax1.twinx()
    ax2.plot(Ms, Nfit, 's--', lw=1.0, color='tab:orange', label="Nfit")
    ax2.axhline(NMIN_FIT, color='tab:orange', ls='--', lw=1.0, alpha=0.7)
    ax2.set_ylabel("N puntos en ajuste")
    ax1.set_xlabel("M")
    ax1.set_title("Calidad del ajuste y puntos en el rango")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout(); plt.savefig(f"{out_prefix}_quality_vs_M.png", dpi=220); plt.show()

    # Linealidad
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(Ms, mdif, 'o-', lw=1.2)
    plt.axhline(LINEARITY_TOL, color='tab:red', ls='--', lw=1.0)
    plt.xlabel("M"); plt.ylabel("|Δ pendiente (mitades)|")
    plt.title("Prueba de linealidad en la ventana de ajuste")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{out_prefix}_linearity_vs_M.png", dpi=220); plt.show()

def plot_parametrizaciones(frame_id, L, r_base, has_z, x_base, y_base, M_list, out_prefix):
    """
    Remuestrea para todos los M, y grafica:
      - si hay z: XY, x(s/L), y(s/L), r(s/L) desde z
      - si no hay z: r(s/L) desde r_al
    """
    curves = []
    for M in M_list:
        ent = {"M": M}
        s = np.linspace(0, L, M, endpoint=False)
        ent["s_norm"] = s / L
        ent["r"] = periodic_resample_1d(r_base, M)
        if has_z:
            xM = periodic_resample_1d(x_base, M)
            yM = periodic_resample_1d(y_base, M)
            ent["x"], ent["y"] = xM, yM
        curves.append(ent)

    if has_z:
        # XY
        plt.figure(figsize=(6.2, 6.2))
        for ent, col in zip(curves, plt.cm.viridis(np.linspace(0,1,len(curves)))):
            plt.plot(ent["x"], ent["y"], '-', lw=0.9, color=col, label=f"M={ent['M']}")
        plt.axis('equal'); plt.gca().invert_yaxis()
        plt.title(f"Parametrizaciones en XY — frame {frame_id}")
        plt.legend(fontsize=8, ncol=2, frameon=False)
        plt.tight_layout(); plt.savefig(f"{out_prefix}_param_XY.png", dpi=220); plt.show()

        # x(s/L) e y(s/L)
        plt.figure(figsize=(9.5, 3.6))
        plt.subplot(1,2,1)
        for ent in curves: plt.plot(ent["s_norm"], ent["x"], lw=0.9, label=f"M={ent['M']}")
        plt.xlabel("s/L"); plt.ylabel("x(s)"); plt.title("x(s/L) por M"); plt.grid(True, alpha=0.3)
        plt.subplot(1,2,2)
        for ent in curves: plt.plot(ent["s_norm"], ent["y"], lw=0.9, label=f"M={ent['M']}")
        plt.xlabel("s/L"); plt.ylabel("y(s)"); plt.title("y(s/L) por M"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{out_prefix}_param_xys_vs_snorm.png", dpi=220); plt.show()

        # r(s/L) desde z (centro promedio de x,y remuestreados)
        plt.figure(figsize=(6.8, 4.0))
        xc = np.mean([np.mean(ent["x"]) for ent in curves])
        yc = np.mean([np.mean(ent["y"]) for ent in curves])
        for ent in curves:
            r_from_z = np.hypot(ent["x"] - xc, ent["y"] - yc)
            plt.plot(ent["s_norm"], r_from_z, lw=0.9, label=f"M={ent['M']}")
        plt.xlabel("s/L"); plt.ylabel("r(s)")
        plt.title(f"r(s/L) desde z(s) — frame {frame_id}")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.legend(fontsize=8, ncol=2, frameon=False)
        plt.savefig(f"{out_prefix}_param_r_from_z_vs_snorm.png", dpi=220); plt.show()
    else:
        # r(s/L) desde r_al
        plt.figure(figsize=(6.8, 4.0))
        for ent in curves:
            plt.plot(ent["s_norm"], ent["r"], lw=0.9, label=f"M={ent['M']}")
        plt.xlabel("s/L"); plt.ylabel("r(s)")
        plt.title(f"r(s/L) por M — frame {frame_id}")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.legend(fontsize=8, ncol=2, frameon=False)
        plt.savefig(f"{out_prefix}_param_r_vs_snorm.png", dpi=220); plt.show()

# ============================== MAIN =========================================

if __name__ == "__main__":

    # Carga datos + frame del medio
    frame_id, L, r_base, has_z, x_base, y_base = load_npz_and_pick_middle_frame(NPZ_PATH)

    # 1) Barrido de M → B(r), xi, etc.
    if RUN_SWEEP_B:
        results = []
        for M in M_CANDIDATES:
            res = compute_B_and_fit_for_M(
                L=L, r_base=r_base, M=M,
                rmin_req=RMIN_REQ, rmax_req=RMAX_REQ,
                min_pts_fit=MIN_PTS_FIT, ds_safety=DS_SAFETY,
                use_auto_window=USE_AUTO_WINDOW,
                auto_min_pts=AUTO_MIN_PTS, auto_min_log_span=AUTO_MIN_LOG_SPAN
            )
            if not res["ok"]:
                print(f"[WARN] M={M}: salteado ({res['reason']}). "
                      f"r∈[{res['rmin_eff']:.2f},{res['rmax_eff']:.2f}] px")
            results.append(res)

        # Plot multi-M de B(r)
        plot_B_multiM(frame_id, L, results, OUT_PREFIX, RMIN_REQ, RMAX_REQ)

        # Guardar CSV resumen (UTF-8 con BOM para Excel en Windows)
        with open(f"{OUT_PREFIX}_resumen.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            # Usar encabezados ASCII
            w.writerow(["frame","L_px","M","xi","B0","R2","Nfit","abs_dm","rmin_fit_px","rmax_fit_px"])
            for r in results:
                if r["ok"]:
                    w.writerow([frame_id, f"{L:.6f}", r["M"],
                                f"{r['xi']:.8f}", f"{r['B0']:.8g}", f"{r['R2']:.6f}",
                                r["Nfit"], f"{r['m_diff']:.6f}",
                                f"{r['rmin_used']:.6f}", f"{r['rmax_used']:.6f}"])
                else:
                    w.writerow([frame_id, f"{L:.6f}", r["M"], "", "", "", 0, "", "", ""])


    # 2) Selección de M “óptimo” + gráficas de calidad
    if RUN_SELECT_M_OPTIMO:
        # Si no corriste RUN_SWEEP_B, calculamos results acá:
        if not RUN_SWEEP_B:
            results = [compute_B_and_fit_for_M(L, r_base, M, RMIN_REQ, RMAX_REQ,
                                               MIN_PTS_FIT, DS_SAFETY,
                                               USE_AUTO_WINDOW, AUTO_MIN_PTS, AUTO_MIN_LOG_SPAN)
                       for M in M_CANDIDATES]

        # Criterios prácticos para elegir M (duros + estabilidad de xi)
        def passes(d):
            return (d["ok"] and d["Nfit"] >= NMIN_FIT and d["R2"] >= R2_MIN and d["m_diff"] <= LINEARITY_TOL)

        valid_idx = [i for i, d in enumerate(results) if passes(d)]
        M_opt = None
        if valid_idx:
            for i in valid_idx:
                xi_i = results[i]["xi"]
                stable = any(results[j]["M"] > results[i]["M"] and
                             abs(results[j]["xi"] - xi_i) <= STABILITY_TOL
                             for j in valid_idx)
                if stable:
                    M_opt = results[i]["M"]; break
            if M_opt is None:  # si no hay par estable, tomamos el menor que pasa filtros duros
                M_opt = results[valid_idx[0]]["M"]

        # Reporte en consola
        print("\n=== RESUMEN POR M ===")
        print(" M    ds(px)  Nfit   xi        R2     |Δm|    rmin_fit  rmax_fit")
        for d in results:
            xi_s  = "NaN" if not d["ok"] else f"{d['xi']:+.5f}"
            R2_s  = "NaN" if not d["ok"] else f"{d['R2']:.3f}"
            md_s  = "NaN" if not d["ok"] else f"{d['m_diff']:.3f}"
            rm_s  = f"{(d.get('rmin_used',d.get('rmin_eff',np.nan))):.2f}"
            rM_s  = f"{(d.get('rmax_used',d.get('rmax_eff',np.nan))):.2f}"
            print(f"{d['M']:5d}  {d['ds']:.3f}  {d.get('Nfit',0):4d}  {xi_s:>8}  {R2_s:>6}  {md_s:>6}  {rm_s}   {rM_s}")

        if M_opt is None:
            print("\n[WARN] Ningún M cumple todos los criterios. "
                  "Aumentá M o relajá R2_MIN/NMIN_FIT/LINEARITY_TOL.")
        else:
            print(f"\n>>> M óptimo sugerido: {M_opt} <<<")

        # Plots de calidad
        plot_quality_vs_M(frame_id, L, results, OUT_PREFIX, mark_x=M_opt)

    # 3) Gráficas de parametrización (XY, x/y/r vs s/L)
    if RUN_PLOT_PARAMETROS:
        plot_parametrizaciones(frame_id, L, r_base, has_z, x_base, y_base, M_CANDIDATES, OUT_PREFIX)

