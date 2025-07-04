# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:54:56 2025

@author: UBA-BT
"""


# -*- coding: utf-8 -*-
"""
Script definitivo para análisis de contornos en imágenes

Funcionalidades:
1. Procesamiento automático de series de imágenes
2. Cálculo de métricas clave (área, perímetro, desplazamiento)
3. Análisis de sensibilidad al suavizado (usando la última imagen)
4. Visualizaciones profesionales
5. Sistema de guardado optimizado
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from scipy.spatial import ConvexHull
import pandas as pd
from bordes_poco_contraste import ImageEnhancer

# Configuración de estilos para gráficos
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'

class ContourAnalysis:
    def __init__(self, image_dir, crop_region, output_dir="analysis_results", start_from=0):
        """Inicializa el analizador con rutas y parámetros básicos"""
        self.image_dir = image_dir
        self.crop_region = crop_region
        self.output_dir = output_dir
        self.start_from = start_from  # Nuevo parámetro
        self.results = {}
        self._setup_directories()
    
    def _setup_directories(self):
        """Crea la estructura de directorios para resultados"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.binary_dir = os.path.join(self.output_dir, "binary_images")
        self.contour_dir = os.path.join(self.output_dir, "contour_overlays")
        os.makedirs(self.binary_dir, exist_ok=True)
        os.makedirs(self.contour_dir, exist_ok=True)
    
    def load_images(self):
        """Carga imágenes ordenadas cronológicamente, empezando desde start_from"""
        pattern = r'resta_(\d{8}_\d{6})\.tif'
        files = []
        
        for f in os.listdir(self.image_dir):
            match = re.match(pattern, f)
            if match:
                timestamp = match.group(1)
                files.append((timestamp, f))
        
        # Ordenar por timestamp y aplicar el filtro start_from
        files.sort()
        files = files[self.start_from:]  # Aplicar el corte aquí
        
        self.last_image = files[-1][1] if files else None
        
        # Cargar las imágenes filtradas
        images = []
        filenames = []
        for ts, fname in files:
            try:
                img = imread(os.path.join(self.image_dir, fname))[self.crop_region]
                images.append(img)
                filenames.append(fname)
            except Exception as e:
                print(f"Error cargando {fname}: {str(e)}")
        
        return images, filenames
    
    def process_images(self, **processing_params):
        """Procesa todas las imágenes con los parámetros dados"""
        images, filenames = self.load_images()
        
        for img, fname in zip(images, filenames):
            try:
                print(f"Procesando: {fname}")
                
                enhancer = ImageEnhancer(img)
                binary, contours, _ = enhancer.procesar(
                    mostrar=False,
                    **processing_params
                )
                
                # Calcular métricas básicas
                area = sum(self._contour_area(c) for c in contours if len(c) >= 3)
                perimeter = sum(self._contour_perimeter(c) for c in contours if len(c) >= 2)
                
                self.results[fname] = {
                    'binary': binary,
                    'contours': contours,
                    'area': area,
                    'perimeter': perimeter,
                    'num_contours': len(contours)
                }
                
                # Guardar imágenes
                self._save_images(fname, binary, contours)
                
            except Exception as e:
                print(f"Error procesando {fname}: {str(e)}")
                self.results[fname] = {'error': str(e)}
    
    def _save_images(self, fname, binary, contours):
        """Guarda las imágenes procesadas"""
        # Guardar binaria
        imsave(os.path.join(self.binary_dir, fname), binary)
        
        # Guardar con contornos
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
    
    @staticmethod
    def _contour_area(contour):
        """Calcula área usando fórmula del shoelace"""
        x, y = contour[:, 1], contour[:, 0]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    @staticmethod
    def _contour_perimeter(contour):
        """Calcula perímetro sumando distancias entre puntos"""
        diffs = np.diff(contour, axis=0, prepend=contour[-1:])
        return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
    
    def analyze_smoothing_sensitivity(self, smoothing_range=(1, 50, 1)):
        """Analiza sensibilidad al parámetro de suavizado"""
        if not self.last_image:
            print("No hay imágenes disponibles para análisis")
            return None, None
        
        print(f"\nAnalizando sensibilidad usando: {self.last_image}")
        
        # Cargar última imagen
        img_path = os.path.join(self.image_dir, self.last_image)
        img = imread(img_path)[self.crop_region]
        
        # Probar diferentes valores de suavizado
        smoothing_values = np.arange(*smoothing_range)
        areas = []
        perimeters = []
        
        for s in smoothing_values:
            enhancer = ImageEnhancer(img)
            _, contours, _ = enhancer.procesar(
                suavizado=s,
                percentil_contornos=99.9,
                min_dist_picos=8000,
                metodo_contorno="binarizacion",
                mostrar=False
            )
            
            area = sum(self._contour_area(c) for c in contours if len(c) >= 3)
            perimeter = sum(self._contour_perimeter(c) for c in contours if len(c) >= 2)
            
            areas.append(area)
            perimeters.append(perimeter)
        
        return smoothing_values, areas, perimeters
    
    def calculate_displacement(self):
        """Calcula la métrica de desplazamiento entre frames"""
        sorted_results = sorted(
            [r for r in self.results.values() if 'error' not in r],
            key=lambda x: x.get('timestamp', '')
        )
        
        displacements = []
        for i in range(len(sorted_results)-1):
            A1, A2 = sorted_results[i]['area'], sorted_results[i+1]['area']
            P1, P2 = sorted_results[i]['perimeter'], sorted_results[i+1]['perimeter']
            P_avg = (P1 + P2) / 2
            displacements.append((A2 - A1) / P_avg if P_avg > 0 else 0)
        
        return np.array(displacements)
    
    def save_results(self):
        """Guarda todas las métricas en un CSV"""
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
        """Genera gráficos profesionales de los resultados"""
        valid_results = [r for r in self.results.values() if 'error' not in r]
        if not valid_results:
            print("No hay datos válidos para graficar")
            return
        
        # Preparar datos
        frames = range(len(valid_results))
        areas = [r['area'] for r in valid_results]
        perimeters = [r['perimeter'] for r in valid_results]
        displacements = self.calculate_displacement()
        
        # Crear figura
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Gráfico de Áreas
        ax1.plot(frames, areas, 'o-', color='#1f77b4', linewidth=1.5, markersize=4)
        ax1.set_title('Evolución del Área', pad=10)
        ax1.set_ylabel('Área (pix²)')
        
        # Gráfico de Perímetros
        ax2.plot(frames, perimeters, 's-', color='#ff7f0e', linewidth=1.5, markersize=4)
        ax2.set_title('Evolución del Perímetro', pad=10)
        ax2.set_ylabel('Perímetro (pix)')
        
        # Gráfico de Desplazamiento
        if len(displacements) > 0:
            ax3.plot(frames[1:], displacements, 'D-', color='#2ca02c', linewidth=1.5, markersize=4)
            ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax3.set_title('Métrica de Desplazamiento (ΔÁrea/Perímetro)', pad=10)
            ax3.set_ylabel('Desplazamiento (pix)')
            ax3.set_xlabel('Número de Frame')
        
        # Ajustes comunes
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "metrics_evolution.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_sensitivity(self, smoothing_values, areas, perimeters):
        """Grafica los resultados del análisis de sensibilidad"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Gráfico de Áreas
        ax1.plot(smoothing_values, areas, 'o-', color='#1f77b4')
        ax1.set_title('Sensibilidad del Área al Suavizado', pad=10)
        ax1.set_ylabel('Área (pix²)')
        
        # Gráfico de Perímetros
        ax2.plot(smoothing_values, perimeters, 's-', color='#ff7f0e')
        ax2.set_title('Sensibilidad del Perímetro al Suavizado', pad=10)
        ax2.set_ylabel('Perímetro (pix)')
        ax2.set_xlabel('Valor de Suavizado')
        
        for ax in [ax1, ax2]:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "sensitivity_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()


def main():
    # =============================================
    # CONFIGURACIÓN PRINCIPAL (EDITAR ESTOS VALORES)
    # =============================================
    IMAGE_DIR = r"D:\Labos 6-7 2025\Baut+Toto\23-06-2025\260 x 5 relocated"
    CROP_REGION = (slice(570, 710), slice(830, 970))  # (filas, columnas)
    START_FROM=11
    # Parámetros de procesamiento
    PROCESSING_PARAMS = {
        'suavizado': 3,
        'percentil_contornos': 99.9,
        'min_dist_picos': 8000,
        'metodo_contorno': "binarizacion"
    }
    
    # Rango para análisis de sensibilidad (inicio, fin, paso)
    SMOOTHING_RANGE = (1, 50, 1)
    
    # =============================================
    # EJECUCIÓN DEL ANÁLISIS
    # =============================================
    print("Iniciando análisis de contornos...")
    analyzer = ContourAnalysis(IMAGE_DIR, CROP_REGION,start_from=START_FROM)
    
    # 1. Procesar todas las imágenes
    analyzer.process_images(**PROCESSING_PARAMS)
    
    # 2. Análisis de sensibilidad (automático con última imagen)
    smoothing_values, areas, perimeters = analyzer.analyze_smoothing_sensitivity(SMOOTHING_RANGE)
    if smoothing_values is not None:
        analyzer.plot_sensitivity(smoothing_values, areas, perimeters)
    
    # 3. Guardar resultados y generar gráficos
    analyzer.save_results()
    analyzer.plot_metrics()
    
    print("\nAnálisis completado exitosamente")


if __name__ == "__main__":
    main()