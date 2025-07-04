# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 11:12:27 2025

@author: Bautista
"""

# -*- coding: utf-8 -*-
"""
Análisis de contornos y métricas morfológicas en imágenes (binarizadas o no)

Funcionalidades:
- Procesamiento de imágenes (binarizadas o con realce previo)
- Cálculo de área, perímetro y desplazamiento
- Visualización y guardado de resultados
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.measure import find_contours
import pandas as pd

# Si se trabaja con imágenes no binarizadas
try:
    from bordes_poco_contraste import ImageEnhancer
except ImportError:
    ImageEnhancer = None


class ContourAnalysis:
    def __init__(self, image_dir, crop_region, output_dir, start_from=0,
                 already_binarized=True, processing_params=None,
                 filename_pattern=r'(Bin-P8139-190Oe-30ms-5Tw-\d+)\.tif'):
        
        self.image_dir = image_dir
        self.crop_region = crop_region
        self.output_dir = output_dir
        self.start_from = start_from
        self.already_binarized = already_binarized
        self.processing_params = processing_params or {}
        self.filename_pattern = filename_pattern
        self.results = {}
        self.last_image = None
        self._setup_directories()
    def _setup_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.binary_dir = os.path.join(self.output_dir, "binary_images")
        self.contour_dir = os.path.join(self.output_dir, "contour_overlays")
        os.makedirs(self.binary_dir, exist_ok=True)
        os.makedirs(self.contour_dir, exist_ok=True)

    def load_images(self):
        pattern = self.filename_pattern
        files = []
    
        for f in os.listdir(self.image_dir):
            match = re.match(pattern, f)
            if match:
                key = match.group(1)
    
                # Intentar detectar si key es un número
                if re.fullmatch(r'\d+', key):
                    sort_key = int(key)
                elif re.fullmatch(r'\d{8}_\d{6}', key):  # timestamp tipo 20250704_101230
                    sort_key = key
                else:
                    # Último recurso: buscar número al final
                    num_match = re.search(r'(\d+)(?=\.\w+$)', f)
                    sort_key = int(num_match.group(1)) if num_match else f
    
                files.append((sort_key, f))
    
        files.sort(key=lambda x: x[0])
        files = files[self.start_from:]
        self.last_image = files[-1][1] if files else None
    
        images, filenames = [], []
        for _, fname in files:
            try:
                img = imread(os.path.join(self.image_dir, fname))[self.crop_region]
                images.append(img)
                filenames.append(fname)
            except Exception as e:
                print(f"Error cargando {fname}: {str(e)}")
    
        return images, filenames


    def process_images(self):
        images, filenames = self.load_images()

        for img, fname in zip(images, filenames):
            try:
                print(f"Procesando: {fname}")
                
                if self.already_binarized:
                    binary = img > 0
                else:
                    enhancer = ImageEnhancer(img)
                    binary, _, _ = enhancer.procesar(
                        mostrar=False,
                        **self.processing_params
                    )
                
                contours = find_contours(binary.astype(float), level=0.5)

                area = sum(self._contour_area(c) for c in contours if len(c) >= 3)
                perimeter = sum(self._contour_perimeter(c) for c in contours if len(c) >= 2)

                self.results[fname] = {
                    'binary': binary,
                    'contours': contours,
                    'area': area,
                    'perimeter': perimeter,
                    'num_contours': len(contours),
                    'filename': fname 
                }

                self._save_images(fname, binary, contours)

            except Exception as e:
                print(f"Error procesando {fname}: {str(e)}")
                self.results[fname] = {'error': str(e)}

    def _save_images(self, fname, binary, contours):
        imsave(os.path.join(self.binary_dir, fname), binary.astype(np.uint8) * 255)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(binary, cmap='gray')
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], linewidth=1, color='cyan')
        ax.axis('off')
        plt.savefig(
            os.path.join(self.contour_dir, f"{os.path.splitext(fname)[0]}.png"),
            dpi=150, bbox_inches='tight', pad_inches=0.1
        )
        plt.close()

    def _contour_area(self, contour):
        x, y = contour[:, 1], contour[:, 0]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _contour_perimeter(self, contour):
        diffs = np.diff(contour, axis=0, prepend=contour[-1:])
        return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

    def calculate_displacement(self):
        sorted_results = sorted(
            [r for r in self.results.values() if 'error' not in r],
            key=lambda x: x.get('filename', '')
        )
        displacements = []
        for i in range(len(sorted_results) - 1):
            A1, A2 = sorted_results[i]['area'], sorted_results[i+1]['area']
            P1, P2 = sorted_results[i]['perimeter'], sorted_results[i+1]['perimeter']
            P_avg = (P1 + P2) / 2
            displacements.append((A2 - A1) / P_avg if P_avg > 0 else 0)
        return np.array(displacements)

    def save_results(self):
        data = []
        for fname, res in self.results.items():
            if 'error' in res:
                continue
            data.append({
                'filename': fname,
                'area': res['area'],
                'perimeter': res['perimeter'],
                'num_contours': res['num_contours']
            })
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.output_dir, "metrics_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nMétricas guardadas en: {csv_path}")

    def plot_metrics(self):
        valid_results = [r for r in self.results.values() if 'error' not in r]
        if not valid_results:
            print("No hay datos válidos para graficar")
            return

        frames = range(len(valid_results))
        areas = [r['area'] for r in valid_results]
        perimeters = [r['perimeter'] for r in valid_results]
        displacements = self.calculate_displacement()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        ax1.plot(frames, areas, 'o-', color='#1f77b4')
        ax1.set_title('Área')
        ax1.set_ylabel('pix²')

        ax2.plot(frames, perimeters, 's-', color='#ff7f0e')
        ax2.set_title('Perímetro')
        ax2.set_ylabel('pix')

        if len(displacements) > 0:
            ax3.plot(frames[1:], displacements, 'D-', color='#2ca02c')
            ax3.axhline(0, color='gray', linestyle='--')
            ax3.set_title('Desplazamiento (ΔÁrea / Perímetro)')
            ax3.set_ylabel('pix')
            ax3.set_xlabel('Frame')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "metrics_evolution.png"), dpi=150)
        plt.show()


def main():
    # =========================================
    # CONFIGURACIÓN
    # =========================================
    #IMAGE_DIR = r"E:\Documents\Labo 6\Fotos campos de velocidades"
    IMAGE_DIR= r'E:\Documents\Labo 6\23-06-2025\260 x 5 relocated'
    OUTPUT_DIR = r"E:\Documents\Labo 6\LuCam-app\analisis\analysis_results_260oe_5ms"
    CROP_REGION = (slice(570,710),slice(830,970))
    #CROP_REGION = (slice(None), slice(None))# Recorte total o ajustá manualmente
    START_FROM = 12
    ALREADY_BINARIZED = False  # Cambiar a False si las imágenes NO están binarizadas
    #FILENAME_PATTERN = r'(Bin-P8139-190Oe-30ms-5Tw-\d+)\.tif'
    FILENAME_PATTERN = r'resta_(\d{8}_\d{6})\.tif'

    # Solo se usan si ALREADY_BINARIZED = False
    PROCESSING_PARAMS = {
        'suavizado': 3,
        'percentil_contornos': 99.9,
        'min_dist_picos': 8000,
        'metodo_contorno': "binarizacion"
    }

    # =========================================
    # EJECUCIÓN
    # =========================================
    print("Iniciando análisis...")
    analyzer = ContourAnalysis(
        image_dir=IMAGE_DIR,
        crop_region=CROP_REGION,
        output_dir=OUTPUT_DIR,
        start_from=START_FROM,
        already_binarized=ALREADY_BINARIZED,
        processing_params=PROCESSING_PARAMS,
        filename_pattern=FILENAME_PATTERN
    )
    analyzer.process_images()
    analyzer.save_results()
    analyzer.plot_metrics()

    print("Análisis finalizado.")

if __name__ == "__main__":
    main()
