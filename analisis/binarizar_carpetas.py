# -*- coding: utf-8 -*-
"""
Binarización por lote usando la clase ImageEnhancer del script 'bordes_poco_contraste.py',
con selección de ROI manual (PyQt5 + QRubberBand) o ROI fijo, y elección de la imagen
en la que se abre el selector (foto_idx_roi o nombre_img_roi).

Requisitos: numpy, scipy, scikit-image, PyQt5
"""

import os, glob, re, sys
import numpy as np

from skimage import io
from skimage.filters import gaussian, threshold_otsu, sobel
from skimage.measure import find_contours
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


# =============================================================================
# CLASE: ImageEnhancer (misma API/algoritmo que en 'bordes_poco_contraste.py')
# =============================================================================

class ImageEnhancer:
    def __init__(self, imagen, sigma_background=100, alpha=0):
        self.image = imagen
        self.sigma_background = sigma_background
        self.alpha = alpha

    def _subtract_background(self):
        background = gaussian(self.image.astype(np.float32),
                              sigma=self.sigma_background,
                              preserve_range=True)
        corrected = self.image.astype(np.float32) - self.alpha * background
        return corrected

    def _detect_histogram_peaks(self, image, min_intensity=5, min_dist=30, usar_dos_picos=True):
        histograma, bins = np.histogram(image[image > min_intensity],
                                        bins=32767*2, range=(0, 32767*2))
        histograma[:5] = 0
        hist = gaussian(histograma.astype(float), sigma=400)
        peaks, _ = find_peaks(hist, distance=min_dist)
        peak_vals = hist[peaks]

        if len(peaks) >= 2 and usar_dos_picos:
            sorted_indices = np.argsort(peak_vals)[-2:]
            top_peaks = peaks[sorted_indices]
            top_peaks.sort()
            centro = (top_peaks[0] + top_peaks[1]) / 2
            sigma = abs(top_peaks[0] - centro)
        elif len(peaks) >= 1:
            mu = peaks[np.argmax(peak_vals)]

            def gauss(x, A, sigma):
                return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

            x_data = np.linspace(0, 32767*2, 32767*2)
            y_data = hist
            try:
                popt, _ = curve_fit(gauss, x_data, y_data, p0=[hist[mu], 10])
                A, sigma = popt
                centro = mu
            except RuntimeError:
                centro = mu
                sigma = 10
            top_peaks = np.array([mu])
        else:
            raise ValueError("No se detectaron picos en el histograma.")

        return centro, sigma, hist, top_peaks

    def _enhance_tanh_diff2(self, corrected, centro, sigma):
        delta = corrected - centro
        return np.exp(-0.5 * (delta / sigma) ** 2) * delta

    def _apply_tanh(self, image, ganancia=1, centro=100, sigma=50):
        delta = image - centro
        return 0.5 * (np.tanh(0.5 * delta / sigma) + 1)

    def _find_large_contours(self, binary, percentil_contornos=0):
        contours = find_contours(binary, level=0.5)
        if percentil_contornos > 0:
            def area_contorno(contour):
                x = contour[:, 1]
                y = contour[:, 0]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas = np.array([area_contorno(c) for c in contours])
            umbral = np.percentile(areas, percentil_contornos)
            return [c for c, a in zip(contours, areas) if a >= umbral]
        return contours

    def _find_contours_by_sobel(self, image, levels=[0.1], percentil_contornos=0):
        edges = sobel(image.astype(float) / 65534)
        contornos = []
        for nivel in levels:
            c = find_contours(edges, level=nivel)
            contornos.extend(c)
        if percentil_contornos > 0 and contornos:
            def area_contorno(contour):
                x = contour[:, 1]
                y = contour[:, 0]
                return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas = np.array([area_contorno(c) for c in contornos])
            umbral = np.percentile(areas, percentil_contornos)
            contornos = [c for c, a in zip(contornos, areas) if a >= umbral]
        return contornos

    def procesar(self, suavizado=5, ganancia_tanh=0.1, mostrar=True,
                 percentil_contornos=0, min_dist_picos=30,
                 metodo_contorno="sobel", usar_dos_picos=True):

        corrected = self._subtract_background()
        centro, sigma, hist, top_peaks = self._detect_histogram_peaks(
            corrected, min_dist=min_dist_picos, usar_dos_picos=usar_dos_picos)

        enhanced = self._enhance_tanh_diff2(corrected, centro, sigma)
        enhanced_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-12)
        enhanced_uint8 = (enhanced_norm * 65534).astype(np.uint16)

        smooth = uniform_filter(enhanced_uint8, size=suavizado)
        centro1, sigma1, hist1, top_peaks1 = self._detect_histogram_peaks(
            smooth, min_dist=min_dist_picos, usar_dos_picos=usar_dos_picos)

        enhanced2 = self._apply_tanh(smooth, ganancia=ganancia_tanh, centro=centro1, sigma=sigma1)
        enhanced2_norm = (enhanced2 - enhanced2.min()) / (enhanced2.max() - enhanced2.min() + 1e-12)
        enhanced2_uint8 = (enhanced2_norm * 65534).astype(np.uint16)

        threshold = threshold_otsu(enhanced2_uint8)
        print("[ImageEnhancer] Otsu threshold:", threshold)
        binary = (enhanced2_uint8 > threshold).astype(np.uint16) * 65534

        if metodo_contorno == "sobel":
            sobel_image = sobel(enhanced2_uint8.astype(float) / 65534)
            contornos = self._find_contours_by_sobel(enhanced2_uint8, levels=[0.16],
                                                     percentil_contornos=percentil_contornos)
            imagen_contorno = sobel_image
        elif metodo_contorno == "binarizacion":
            contornos = self._find_large_contours(binary, percentil_contornos=percentil_contornos)
            imagen_contorno = binary
        else:
            raise ValueError(f"Método de contorno no reconocido: {metodo_contorno}")

        if mostrar:
            import matplotlib.pyplot as plt
            self._mostrar_resultados(enhanced_uint8, smooth, enhanced2_uint8, binary,
                                     contornos, hist, top_peaks, threshold, imagen_contorno, plt)

        return binary, contornos, hist

    def _mostrar_resultados(self, enhanced_uint8, smooth, enhanced2_uint8, binary,
                            contornos, hist, top_peaks, threshold, imagen_contorno, plt):
        plt.figure(figsize=(18, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(self.image, cmap='gray')
        for c in contornos:
            plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
        plt.title("Original + contornos")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(enhanced_uint8, cmap='gray')
        plt.title("Realce (x*exp(-0.5(x/sigma)**2))")
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(smooth, cmap='gray')
        plt.title("Suavizado")
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(enhanced2_uint8, cmap='gray')
        plt.title("Tanh final")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(imagen_contorno, cmap='gray')
        for c in contornos:
            plt.scatter(c[:, 1], c[:, 0], s=1, c='cyan')
        plt.title("Contorno sobre imagen de referencia")
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.plot(hist, color='gray')
        plt.scatter(top_peaks, hist[top_peaks], color='red')
        plt.title("Histograma + Picos seleccionados")
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")

        print(f"Cantidad de contornos detectados: {len(contornos)}")
        plt.tight_layout()
        plt.show()


# =============================================================================
# Selector de ROI (PyQt5 puro, con debug y orden correcto de QApplication)
# =============================================================================

def _natsort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def seleccionar_roi_qt(imagen_2d, titulo="Seleccionar ROI"):
    """
    Selector de ROI con PyQt5 + QRubberBand.
    Devuelve (r0,r1,c0,c1) o None si se cancela.
    """
    print("\n[ROI-Qt] >>> Entrando a seleccionar_roi_qt()")
    print(f"[ROI-Qt] título = {titulo!r}")
    print(f"[ROI-Qt] tipo(imagen_2d) = {type(imagen_2d)}")

    if hasattr(imagen_2d, "shape"):
        print(f"[ROI-Qt] shape = {imagen_2d.shape}, dtype = {imagen_2d.dtype}")
    else:
        print("[ROI-Qt] Advertencia: imagen_2d no tiene atributo 'shape'")

    # Importar PyQt5
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        print("[ROI-Qt] PyQt5 importado correctamente.")
    except Exception as e:
        print("[ROI-Qt] No se pudo importar PyQt5:", e)
        return None

    # Crear/obtener QApplication ANTES de tocar QPixmap
    app = QtWidgets.QApplication.instance()
    if app is None:
        print("[ROI-Qt] No existía QApplication, creando una nueva.")
        app = QtWidgets.QApplication(sys.argv or [])
    else:
        print("[ROI-Qt] Ya había una QApplication activa.")

    # Validar la imagen de entrada
    if imagen_2d is None:
        print("[ROI-Qt] Error: imagen_2d es None.")
        return None
    if imagen_2d.size == 0:
        print("[ROI-Qt] Error: La imagen de entrada está vacía (size=0).")
        return None
    if not np.any(imagen_2d):
        print("[ROI-Qt] Advertencia: La imagen para seleccionar el ROI es completamente negra/cero.")

    img = imagen_2d.astype(np.float32)
    try:
        vmin = float(np.min(img))
        vmax = float(np.max(img))
        vmean = float(np.mean(img))
        print(f"[ROI-Qt] Estadísticos imagen: min={vmin}, max={vmax}, mean={vmean}")
    except Exception as e:
        print("[ROI-Qt] Error calculando min/max/mean:", e)

    try:
        p1, p99 = np.percentile(img, [1, 99])
        print(f"[ROI-Qt] Percentiles 1/99: p1={p1}, p99={p99}")
    except Exception as e:
        print("[ROI-Qt] Error calculando percentiles:", e)
        p1, p99 = float(img.min()), float(img.max())

    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        print("[ROI-Qt] Percentiles raros, usando min/max.")
        p1, p99 = float(img.min()), float(img.max())

    if p99 <= p1:
        print("[ROI-Qt] Imagen plana, ajusto p99 = p1 + 1.")
        p99 = p1 + 1.0

    img8 = np.clip((img - p1) / (p99 - p1), 0, 1)
    img8 = (img8 * 255).astype(np.uint8)

    h, w = img8.shape
    print(f"[ROI-Qt] Imagen para GUI: shape={img8.shape}, dtype={img8.dtype}")

    # Crear QImage de forma segura copiando los bytes
    try:
        qimg = QtGui.QImage(w, h, QtGui.QImage.Format_Grayscale8)
        bytes_per_line = qimg.bytesPerLine()
        print(f"[ROI-Qt] QImage creado: bytesPerLine={bytes_per_line}")
        # Adaptamos el size del buffer y copiamos los datos de NumPy
        ptr = qimg.bits()
        ptr.setsize(h * bytes_per_line)
        # Si bytes_per_line == w, simplemente volcamos la imagen row-major
        # Para generalidad, copiamos fila por fila
        arr_flat = img8.ravel()
        if bytes_per_line == w:
            ptr[:h * w] = arr_flat.tobytes()
        else:
            # Copia por filas si hubiera padding
            for row in range(h):
                start_np = row * w
                end_np = start_np + w
                start_qt = row * bytes_per_line
                ptr[start_qt:start_qt + w] = arr_flat[start_np:end_np].tobytes()
        pix = QtGui.QPixmap.fromImage(qimg)
        print("[ROI-Qt] QPixmap creado correctamente.")
    except Exception as e:
        print("[ROI-Qt] Error creando QImage/QPixmap:", e)
        return None

    class ImageLabel(QtWidgets.QLabel):
        def __init__(self, pixmap):
            super().__init__()
            self.setPixmap(pixmap)
            self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            self.rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
            self.origin = None
            self.sel_rect = None
            self.setMouseTracking(True)

        def mousePressEvent(self, e):
            if e.button() == QtCore.Qt.LeftButton:
                self.origin = e.pos()
                self.rubber.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
                self.rubber.show()

        def mouseMoveEvent(self, e):
            if self.origin is not None:
                rect = QtCore.QRect(self.origin, e.pos()).normalized()
                self.rubber.setGeometry(rect)

        def mouseReleaseEvent(self, e):
            if e.button() == QtCore.Qt.LeftButton and self.origin is not None:
                rect = QtCore.QRect(self.origin, e.pos()).normalized()
                self.sel_rect = rect
                self.rubber.setGeometry(rect)
                self.origin = None

        def get_roi_rc(self):
            if self.sel_rect is None:
                print("[ROI-Qt] get_roi_rc: sel_rect=None")
                return None
            x0 = max(0, min(self.sel_rect.left(),  self.pixmap().width()  - 1))
            y0 = max(0, min(self.sel_rect.top(),   self.pixmap().height() - 1))
            x1 = max(0, min(self.sel_rect.right() + 1,  self.pixmap().width()))
            y1 = max(0, min(self.sel_rect.bottom() + 1, self.pixmap().height()))
            roi_rc = (int(y0), int(y1), int(x0), int(x1))  # (r0,r1,c0,c1)
            print(f"[ROI-Qt] get_roi_rc -> {roi_rc}")
            return roi_rc

    class RoiDialog(QtWidgets.QDialog):
        def __init__(self, pixmap, title):
            super().__init__()
            self.setWindowTitle(title)
            self.resize(min(1000, w + 80), min(800, h + 120))
            vbox = QtWidgets.QVBoxLayout(self)

            self.label = ImageLabel(pixmap)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.label)
            vbox.addWidget(scroll)

            hb = QtWidgets.QHBoxLayout()
            hb.addWidget(QtWidgets.QLabel("Arrastrá para marcar el ROI. Enter/Aceptar para confirmar."))
            hb.addStretch(1)
            self.btn_ok = QtWidgets.QPushButton("Aceptar")
            self.btn_ok.setDefault(True)
            self.btn_cancel = QtWidgets.QPushButton("Cancelar")
            hb.addWidget(self.btn_ok)
            hb.addWidget(self.btn_cancel)
            vbox.addLayout(hb)

            self.btn_ok.clicked.connect(self.accept)
            self.btn_cancel.clicked.connect(self.reject)

        def get_roi(self):
            return self.label.get_roi_rc()

    dlg = RoiDialog(pix, titulo)
    print("[ROI-Qt] Ejecutando diálogo modal dlg.exec_()...")
    result = dlg.exec_()
    print(f"[ROI-Qt] dlg.exec_() devolvió: {result}")

    if result == QtWidgets.QDialog.Accepted:
        roi = dlg.get_roi()
        print(f"[ROI-Qt] ROI devuelto por diálogo: {roi}")
        if roi is None:
            print("[ROI-Qt] No se seleccionó rectángulo (roi=None).")
            return None
        return roi
    else:
        print("[ROI-Qt] Diálogo cancelado por el usuario.")
        return None


# =============================================================================
# Batch principal usando ImageEnhancer (con debug)
# =============================================================================

def binarizar_carpeta(
    carpeta_in,
    carpeta_out,
    patron="*.tif",
    roi=None,                  # (r0,r1,c0,c1) o None
    usar_roi_qt=False,         # True => abre selector Qt (ignora roi fijo)
    foto_idx_roi=1,            # 1-based: en qué imagen abrir el selector
    nombre_img_roi=None,       # substring prioritario para elegir imagen ROI
    orden_natural=True,        # orden natural de archivos
    roi_obligatorio=True,      # si el usuario cancela y True -> raise; si False -> usa imagen completa
    # --- parámetros para ImageEnhancer.procesar ---
    sigma_background=100,
    alpha=0,
    suavizado=5,
    ganancia_tanh=0.1,
    percentil_contornos=0,
    min_dist_picos=30,
    metodo_contorno="binarizacion",   # o "sobel"
    usar_dos_picos=True,
    mostrar=False                      # dejar False para evitar matplotlib
):
    print("\n==============================")
    print("=== binarizar_carpeta()    ===")
    print("==============================")
    print(f"[CFG] carpeta_in     = {carpeta_in}")
    print(f"[CFG] carpeta_out    = {carpeta_out}")
    print(f"[CFG] patron         = {patron}")
    print(f"[CFG] usar_roi_qt    = {usar_roi_qt}")
    print(f"[CFG] roi inicial    = {roi}")
    print(f"[CFG] foto_idx_roi   = {foto_idx_roi}")
    print(f"[CFG] nombre_img_roi = {nombre_img_roi}")
    print(f"[CFG] roi_obligatorio= {roi_obligatorio}")
    print(f"[CFG] orden_natural  = {orden_natural}")
    print(f"[CFG] sigma_background={sigma_background}, alpha={alpha}")
    print(f"[CFG] suavizado={suavizado}, ganancia_tanh={ganancia_tanh}")
    print(f"[CFG] percentil_contornos={percentil_contornos}, min_dist_picos={min_dist_picos}")
    print(f"[CFG] metodo_contorno={metodo_contorno!r}, usar_dos_picos={usar_dos_picos}")

    print(f"[CHK] Existe carpeta_in? {os.path.isdir(carpeta_in)}")
    os.makedirs(carpeta_out, exist_ok=True)
    print(f"[CHK] carpeta_out creada/verificada: {carpeta_out}")

    archivos = glob.glob(os.path.join(carpeta_in, patron))
    print(f"[CHK] Encontré {len(archivos)} archivos con patrón {patron!r}")

    if not archivos:
        raise FileNotFoundError(f"No encontré imágenes con patrón {patron} en {carpeta_in}")

    archivos = sorted(archivos, key=_natsort_key if orden_natural else None)
    print("[CHK] Primeros archivos:")
    for p in archivos[:5]:
        print("   -", os.path.basename(p))

    # ------------------------------
    # Selección de ROI manual
    # ------------------------------
    if usar_roi_qt:
        print("\n[ROI] usar_roi_qt=True, voy a elegir imagen para seleccionar ROI.")
        if nombre_img_roi is not None:
            print(f"[ROI] Buscando archivo que contenga '{nombre_img_roi}'...")
            candidatos = [i for i, p in enumerate(archivos) if nombre_img_roi in os.path.basename(p)]
            print(f"[ROI] Índices candidatos = {candidatos}")
            if not candidatos:
                raise FileNotFoundError(f"No encontré ninguna imagen conteniendo '{nombre_img_roi}'")
            idx = candidatos[0]
        else:
            idx = max(1, min(int(foto_idx_roi), len(archivos))) - 1  # clamp y 0-based
            print(f"[ROI] Usando foto_idx_roi={foto_idx_roi} -> idx={idx} (1-based={idx+1})")

        path_roi = archivos[idx]
        print(f"[ROI] Imagen para ROI: {path_roi}")

        img0 = io.imread(path_roi)
        print(f"[ROI] img0.shape={img0.shape}, dtype={img0.dtype}")
        if img0.ndim > 2:
            print("[ROI] Imagen con canales, me quedo con el canal 0.")
            img0 = img0[..., 0]
        print(f"[ROI] img0 (2D) shape={img0.shape}, dtype={img0.dtype}")

        titulo = f"Seleccionar ROI en imagen #{idx+1}: {os.path.basename(archivos[idx])}"
        roi = seleccionar_roi_qt(img0, titulo=titulo)
        print(f"[ROI] Elegido en #{idx+1}: {roi}")
        if roi is None:
            print(f"[ROI] roi=None y roi_obligatorio={roi_obligatorio}")
            if roi_obligatorio:
                print("[ROI] Abortando por roi_obligatorio=True.")
                raise RuntimeError("Selección de ROI cancelada por el usuario.")
            else:
                roi = (0, img0.shape[0], 0, img0.shape[1])
                print("[ROI] Cancelado, usando imagen completa como ROI:", roi)
    else:
        print("[ROI] usar_roi_qt=False, se usará roi fijo o imagen completa.")

    # ------------------------------
    # Procesar todas las imágenes
    # ------------------------------
    resultados = []
    n_total = len(archivos)
    print("\n[PROC] Comienzo del loop sobre imágenes.")

    for i, path in enumerate(archivos, 1):
        base = os.path.splitext(os.path.basename(path))[0]
        print("\n[PROC] ----------------------------------------")
        print(f"[PROC] Procesando {i}/{n_total}: {path}")
        img = io.imread(path)
        print(f"[PROC] img.shape={img.shape}, dtype={img.dtype}")

        if img.ndim > 2:
            print("[PROC] Imagen con canales, usando canal 0.")
            img = img[..., 0]
        print(f"[PROC] img (2D) shape={img.shape}, dtype={img.dtype}")

        if roi is not None:
            print(f"[PROC] Aplicando ROI {roi}")
            r0, r1, c0, c1 = map(int, roi)
            r0 = max(0, r0); c0 = max(0, c0)
            r1 = min(img.shape[0], r1); c1 = min(img.shape[1], c1)
            print(f"[PROC] ROI clamped -> (r0,r1,c0,c1)=({r0},{r1},{c0},{c1})")
            img_proc = img[r0:r1, c0:c1]
        else:
            print("[PROC] Sin ROI: se usa la imagen completa.")
            img_proc = img

        print(f"[PROC] img_proc.shape={img_proc.shape}, dtype={img_proc.dtype}")

        enhancer = ImageEnhancer(img_proc, sigma_background=sigma_background, alpha=alpha)
        print("[PROC] Llamando a enhancer.procesar(...)")
        binary, contornos, hist = enhancer.procesar(
            suavizado=suavizado,
            ganancia_tanh=ganancia_tanh,
            mostrar=mostrar,
            percentil_contornos=percentil_contornos,
            min_dist_picos=min_dist_picos,
            metodo_contorno=metodo_contorno,
            usar_dos_picos=usar_dos_picos
        )
        print(f"[PROC] enhancer.procesar() ok. n_contornos={len(contornos)}")

        out_path = os.path.join(carpeta_out, f"{base}_bin.tif")
        print(f"[PROC] Guardando {out_path}")
        io.imsave(out_path, binary.astype(np.uint16))

        resultados.append((path, out_path, {"roi": roi, "n_contornos": len(contornos)}))
        print(f"[PROC] DONE: {base} -> {out_path} | contornos={len(contornos)}")

    print("\n[PROC] Terminado binarizar_carpeta().")
    return resultados


# =============================================================================
# Ejemplo mínimo
# =============================================================================
if __name__ == "__main__":
    print("\n[MAIN] >>> Entrando en __main__ de binarizar_carpetas.py")

    IN  = r"C:\Users\usuario\Documents\Labo 67\LuCam-app\Data\Mediciones 27-11\toto_bauti_capos\SEQ_Sat300Oe_100ms_Nuc190Oe_300ms_Grow195Oe_100ms_FULL_20251127_154215\rep_010"
    OUT = os.path.join(IN, "out")

    print(f"[MAIN] IN  = {IN}")
    print(f"[MAIN] OUT = {OUT}")

    # ROI fijo (se ignora si usar_roi_qt=True)
    ROI_FIJO = (200, 800, 300, 1100)

    # ROI manual
    USAR_ROI_QT     = True       # abre selector Qt
    FOTO_IDX_ROI    = 10          # imagen (1-based) donde elegir el ROI
    NOMBRE_IMG_ROI  = None       # o un substring de archivo (prioriza sobre el índice)
    ROI_OBLIGATORIO = True       # si cancelás, aborta; si False, usa imagen completa

    binarizar_carpeta(
        IN, OUT, patron="*.tif",
        roi=ROI_FIJO,
        usar_roi_qt=USAR_ROI_QT,
        foto_idx_roi=FOTO_IDX_ROI,
        nombre_img_roi=NOMBRE_IMG_ROI,
        roi_obligatorio=ROI_OBLIGATORIO,
        orden_natural=True,
        # parámetros de ImageEnhancer / procesar
        suavizado=5,
        percentil_contornos=99.9,
        min_dist_picos=8000,
        metodo_contorno="binarizacion",
        usar_dos_picos=True,
        mostrar=False
    )
