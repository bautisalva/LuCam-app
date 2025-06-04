import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation, colors
from skimage import io, measure
from scipy.interpolate import splprep, splev
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

class ContourAnalyzer:
    def __init__(self, folder_path, delta_t=1.0, n_points=200):
        self.folder_path = folder_path
        self.delta_t = delta_t
        self.n_points = n_points
        self.raw_contours = None
        self.interp_contours = None
        self.velocity_fields = []
        self.normal_velocities = []
        self.tangential_velocities = []
        
        self.load_and_process_contours()
        self.compute_velocity_fields()
        self.compute_component_velocities()
    
    def extract_index(self, fname):
        match = re.search(r'5Tw-(\d+)', fname)
        return int(match.group(1)) if match else -1

    def load_contours(self):
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.tif')]
        files_sorted = sorted(files, key=self.extract_index, reverse=True)
        contours = []
        for fname in files_sorted:
            img = io.imread(os.path.join(self.folder_path, fname))
            found = measure.find_contours(img, level=0.5)
            if found:
                largest = max(found, key=len)
                contours.append(largest)
        return contours

    def interpolate_contour(self, contour):
        x, y = contour[:, 1], contour[:, 0]
        if np.linalg.norm([x[0] - x[-1], y[0] - y[-1]]) > 1:
            x, y = np.append(x, x[0]), np.append(y, y[0])
        tck, _ = splprep([x, y], s=0, per=True)
        u = np.linspace(0, 1, self.n_points)
        return np.array(splev(u, tck)).T

    def load_and_process_contours(self):
        self.raw_contours = self.load_contours()
        self.interp_contours = [self.interpolate_contour(c) for c in self.raw_contours]
    
    def compute_velocity_fields(self):
        for i in range(len(self.interp_contours) - 1):
            v = (self.interp_contours[i + 1] - self.interp_contours[i]) / self.delta_t
            self.velocity_fields.append(v)
    
    def compute_normals(self, contour):
        dx = np.roll(contour[:, 0], -1) - np.roll(contour[:, 0], 1)
        dy = np.roll(contour[:, 1], -1) - np.roll(contour[:, 1], 1)
        tangents = np.stack([dx, dy], axis=1)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents /= norms + 1e-8
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        return normals

    def compute_component_velocities(self):
        for i in range(len(self.interp_contours) - 1):
            normals = self.compute_normals(self.interp_contours[i])
            tangents = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
            
            v_normal = np.sum(self.velocity_fields[i] * normals, axis=1)
            v_tangent = np.sum(self.velocity_fields[i] * tangents, axis=1)
            
            self.normal_velocities.append(v_normal)
            self.tangential_velocities.append(v_tangent)

    # ========================
    # VISUALIZACIONES
    # ========================
    
    def plot_temporal_contours(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Obtener colormap y normalizar los índices
        cmap = plt.colormaps.get_cmap("plasma")
        norm = plt.Normalize(vmin=0, vmax=len(self.interp_contours)-1)
        
        for idx, contour in enumerate(self.interp_contours):
            ax.plot(contour[:, 0], contour[:, 1], 
                    color=cmap(norm(idx)), linewidth=1.5, alpha=0.7)
        
        ax.invert_yaxis()
        ax.set_title("Evolución Temporal de los Contornos")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
    
        # ScalarMappable para la barra de color
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # necesario para evitar warning
        
        cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(0, len(self.interp_contours)-1, 5))
        cbar.set_label("Frame")
        
        plt.tight_layout()
        plt.show()

    
    def animate_contours(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def update(i):
            ax.clear()
            p0, p1 = self.interp_contours[i], self.interp_contours[i + 1]
            v = self.velocity_fields[i]
            
            # Plot contornos
            ax.plot(p0[:, 0], p0[:, 1], 'b-', linewidth=2, alpha=0.7, label=f'Frame {i}')
            ax.plot(p1[:, 0], p1[:, 1], 'g--', linewidth=2, alpha=0.7, label=f'Frame {i+1}')
            
            # Plot vectores velocidad
            ax.quiver(p0[:, 0], p0[:, 1], v[:, 0], v[:, 1], 
                      angles='xy', scale_units='xy', scale=1, 
                      color='r', alpha=0.6, width=0.004)
            
            ax.set_title(f"Evolución del Contorno y Campo de Velocidades (Frame {i} → {i+1})")
            ax.axis('equal')
            ax.invert_yaxis()
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
        
        ani = animation.FuncAnimation(fig, update, frames=len(self.velocity_fields), interval=500)
        plt.show()
        return ani
    
    def plot_velocity_heatmap(self, frame_idx):
        p0 = self.interp_contours[frame_idx]
        speed = np.linalg.norm(self.velocity_fields[frame_idx], axis=1)
        
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(p0[:, 0], p0[:, 1], c=speed, 
                         cmap='viridis', s=50, alpha=0.8,
                         edgecolors='w', linewidth=0.5)
        
        plt.colorbar(sc, label='Magnitud de Velocidad', shrink=0.8)
        plt.title(f"Mapa de Calor de Velocidad (Frame {frame_idx} → {frame_idx+1})")
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()
    
    def plot_velocity_components(self, frame_idx):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Velocidad Normal
        p0 = self.interp_contours[frame_idx]
        v_normal = self.normal_velocities[frame_idx]
        sc1 = ax1.scatter(p0[:, 0], p0[:, 1], c=v_normal, 
                          cmap='bwr', s=50, alpha=0.8,
                          vmin=-np.max(np.abs(v_normal)), 
                          vmax=np.max(np.abs(v_normal)),
                          edgecolors='k', linewidth=0.3)
        
        ax1.set_title(f"Velocidad Normal (Frame {frame_idx} → {frame_idx+1})")
        ax1.axis('equal')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.2)
        fig.colorbar(sc1, ax=ax1, label='Velocidad Normal', shrink=0.7)
        
        # Velocidad Tangencial
        v_tangent = self.tangential_velocities[frame_idx]
        sc2 = ax2.scatter(p0[:, 0], p0[:, 1], c=v_tangent, 
                          cmap='coolwarm', s=50, alpha=0.8,
                          vmin=-np.max(np.abs(v_tangent)), 
                          vmax=np.max(np.abs(v_tangent)),
                          edgecolors='k', linewidth=0.3)
        
        ax2.set_title(f"Velocidad Tangencial (Frame {frame_idx} → {frame_idx+1})")
        ax2.axis('equal')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.2)
        fig.colorbar(sc2, ax=ax2, label='Velocidad Tangencial', shrink=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def plot_velocity_histograms(self, frame_idx):
        speed = np.linalg.norm(self.velocity_fields[frame_idx], axis=1)
        v_normal = self.normal_velocities[frame_idx]
        v_tangent = self.tangential_velocities[frame_idx]
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histograma magnitud
        ax[0].hist(speed, bins=30, color='teal', alpha=0.7, edgecolor='k')
        ax[0].set_title(f"Magnitud de Velocidad (Frame {frame_idx})")
        ax[0].set_xlabel("Velocidad")
        ax[0].set_ylabel("Frecuencia")
        ax[0].grid(True, alpha=0.3)
        
        # Histograma velocidad normal
        ax[1].hist(v_normal, bins=30, color='royalblue', alpha=0.7, edgecolor='k')
        ax[1].set_title(f"Velocidad Normal (Frame {frame_idx})")
        ax[1].set_xlabel("Velocidad Normal")
        ax[1].grid(True, alpha=0.3)
        
        # Histograma velocidad tangencial
        ax[2].hist(v_tangent, bins=30, color='purple', alpha=0.7, edgecolor='k')
        ax[2].set_title(f"Velocidad Tangencial (Frame {frame_idx})")
        ax[2].set_xlabel("Velocidad Tangencial")
        ax[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_temporal_evolution(self):
        mean_speeds = []
        mean_normal = []
        mean_tangential = []
        
        for i in range(len(self.velocity_fields)):
            mean_speeds.append(np.mean(np.linalg.norm(self.velocity_fields[i], axis=1)))
            mean_normal.append(np.mean(self.normal_velocities[i]))
            mean_tangential.append(np.mean(self.tangential_velocities[i]))
        
        frames = np.arange(len(mean_speeds))
        
        plt.figure(figsize=(10, 6))
        plt.plot(frames, mean_speeds, 'o-', color='darkgreen', label='Magnitud Velocidad', linewidth=2)
        plt.plot(frames, mean_normal, 's--', color='royalblue', label='Componente Normal', linewidth=1.5)
        plt.plot(frames, mean_tangential, 'd-.', color='purple', label='Componente Tangencial', linewidth=1.5)
        
        plt.title("Evolución Temporal de las Componentes de Velocidad")
        plt.xlabel("Frame")
        plt.ylabel("Velocidad Promedio")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_cumulative_normal_velocity(self):
        cumulative = np.zeros(self.n_points)
        
        for i in range(len(self.normal_velocities)):
            cumulative += self.normal_velocities[i]
        
        p0 = self.interp_contours[0]
        
        # Crear colormap personalizado
        colors = ["blue", "white", "red"]
        cmap = LinearSegmentedColormap.from_list("custom_rdbu", colors)
        
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(p0[:, 0], p0[:, 1], c=cumulative, 
                         cmap=cmap, s=50, alpha=0.8,
                         edgecolors='k', linewidth=0.5,
                         vmin=-np.max(np.abs(cumulative)), 
                         vmax=np.max(np.abs(cumulative)))
        
        plt.colorbar(sc, label='Avance Normal Acumulado', shrink=0.8)
        plt.title("Mapa de Avance Normal Acumulado")
        plt.axis("equal")
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()
    
    def plot_velocity_distribution(self):
        all_speeds = np.concatenate([np.linalg.norm(v, axis=1) for v in self.velocity_fields])
        
        # Densidad de probabilidad
        kde = gaussian_kde(all_speeds)
        x = np.linspace(0, np.max(all_speeds), 500)
        pdf = kde(x)
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_speeds, bins=50, density=True, color='skyblue', alpha=0.7, edgecolor='k', label='Histograma')
        plt.plot(x, pdf, 'r-', linewidth=2, label='Densidad de Probabilidad')
        
        plt.title("Distribución Global de Velocidades")
        plt.xlabel("Velocidad")
        plt.ylabel("Densidad")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_angular_distribution(self, frame_idx):
        p0 = self.interp_contours[frame_idx]
        centroid = np.mean(p0, axis=0)
        vectors = p0 - centroid
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # [-π, π]
        angles_deg = np.degrees(angles)  # Convertir a grados
        
        # Velocidad normal para colorear
        v_normal = self.normal_velocities[frame_idx]
        
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(angles_deg, v_normal, c=v_normal, 
                         cmap='bwr', s=50, alpha=0.7,
                         vmin=-np.max(np.abs(v_normal)), 
                         vmax=np.max(np.abs(v_normal)))
        
        plt.colorbar(label='Velocidad Normal')
        plt.title(f"Distribución Angular de Velocidad Normal (Frame {frame_idx})")
        plt.xlabel("Ángulo (grados)")
        plt.ylabel("Velocidad Normal")
        plt.xticks(np.arange(-180, 181, 45))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Configuración inicial
    FOLDER_PATH = r"C:\Users\Marina\Documents\Labo 6\imagenes\test_velocidades"
    analyzer = ContourAnalyzer(FOLDER_PATH)
    
    # Análisis y visualizaciones
    analyzer.plot_temporal_contours()
    #analyzer.animate_contours()
    anim = analyzer.animate_contours()
    # Ejemplos para un frame específico
    FRAME_IDX = 2
    analyzer.plot_velocity_heatmap(FRAME_IDX)
    analyzer.plot_velocity_components(FRAME_IDX)
    analyzer.plot_velocity_histograms(FRAME_IDX)
    analyzer.plot_angular_distribution(FRAME_IDX)
    
    # Análisis globales
    analyzer.plot_temporal_evolution()
    analyzer.plot_cumulative_normal_velocity()
    analyzer.plot_velocity_distribution()