"""Script autónomo para binarizar imágenes de resta en secuencias SEQ.

El flujo recorre automáticamente las carpetas SEQ ubicadas dentro de un
directorio base (``root``), identifica las imágenes dentro del subdirectorio de
"resta" (configurable) y aplica el procesamiento definido en
``bordes_poco_contraste.ImageEnhancer``. El resultado binarizado se guarda en un
subdirectorio de salida, conservando la estructura de carpetas original.

Todos los parámetros de entrada se establecen directamente en la sección
``CONFIGURACIÓN`` al final del archivo para evitar dependencias con la línea de
comandos. Ajuste esos valores según su entorno antes de ejecutar el script.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from skimage.io import imread, imsave

from bordes_poco_contraste import ImageEnhancer


LOGGER = logging.getLogger(__name__)


@dataclass
class BinarizationConfig:
    """Parámetros necesarios para ejecutar el proceso de binarización."""

    root: Path
    resta_subdir: str = "resta"
    image_pattern: str = "*.tif*"
    images: Optional[Sequence[str]] = None
    output_dir: Optional[Path] = None
    roi: Optional[Tuple[int, int, int, int]] = None
    sigma_background: float = 100.0
    alpha: float = 0.0
    suavizado: int = 5
    ganancia_tanh: float = 0.1
    percentil_contornos: float = 0.0
    min_dist_picos: int = 30
    metodo_contorno: str = "sobel"
    usar_un_solo_pico: bool = False
    log_level: str = "INFO"


def collect_sequence_dirs(root: Path) -> List[Path]:
    """Devuelve las carpetas SEQ_* presentes en ``root``."""
    if not root.is_dir():
        raise FileNotFoundError(f"El directorio base '{root}' no existe o no es una carpeta.")

    seq_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("SEQ"))
    if not seq_dirs:
        LOGGER.warning("No se encontraron carpetas que comiencen con 'SEQ' en %s", root)
    return seq_dirs


def should_process(path: Path, selected_names: Optional[Sequence[str]]) -> bool:
    """Determina si ``path`` debe procesarse según ``selected_names``."""
    if not selected_names:
        return True

    stem = path.stem
    return any(stem == name or path.name == name for name in selected_names)


def load_image(path: Path) -> np.ndarray:
    """Carga una imagen TIFF y devuelve un arreglo 2D en ``np.uint16``."""
    image = imread(path)
    if image.ndim > 2:
        # Seleccionamos el primer canal si hay más dimensiones (p.ej. RGB o stacks).
        image = image[..., 0]
    image = np.asarray(image, dtype=np.uint16)
    return image


def extract_roi(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Obtiene la ROI de ``image`` y asegura que esté dentro de los límites."""
    if roi is None:
        h, w = image.shape
        return image, (0, 0, w, h)

    x, y, roi_w, roi_h = roi
    y_max = min(y + roi_h, image.shape[0])
    x_max = min(x + roi_w, image.shape[1])
    cropped = image[y:y_max, x:x_max]
    return cropped, (x, y, x_max - x, y_max - y)


def insert_roi(base_shape: Tuple[int, int], roi_coords: Tuple[int, int, int, int], roi_data: np.ndarray) -> np.ndarray:
    """Inserta ``roi_data`` en una imagen del tamaño ``base_shape``."""
    full = np.zeros(base_shape, dtype=roi_data.dtype)
    x, y, w, h = roi_coords
    full[y : y + h, x : x + w] = roi_data
    return full


def ensure_output_dir(base_output: Optional[Path], seq_dir: Path) -> Path:
    """Obtiene o crea el directorio de salida para ``seq_dir``."""
    if base_output is None:
        output_dir = seq_dir / "binarizado"
    else:
        base_output = Path(base_output)
        output_dir = base_output / seq_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def process_image(
    image_path: Path,
    enhancer_kwargs: dict,
    process_kwargs: dict,
    roi: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Procesa una única imagen y devuelve la imagen binarizada completa."""
    original = load_image(image_path)
    roi_image, roi_coords = extract_roi(original, roi)

    enhancer = ImageEnhancer(roi_image, **enhancer_kwargs)
    binary_roi, _, _ = enhancer.procesar(mostrar=False, **process_kwargs)

    # Aseguramos que el resultado sea binario (0 o 65534) y del mismo tipo que la entrada.
    binary_roi = np.asarray(binary_roi, dtype=np.uint16)
    binary_full = insert_roi(original.shape, roi_coords, binary_roi)
    return binary_full


def save_binary_image(output_path: Path, image: np.ndarray) -> None:
    """Guarda ``image`` en ``output_path`` en formato TIFF."""
    imsave(output_path, image, check_contrast=False)


def iter_resta_images(seq_dir: Path, resta_subdir: str, pattern: str) -> Iterable[Path]:
    """Itera sobre las imágenes que coinciden con ``pattern`` en ``resta_subdir``."""
    resta_dir = seq_dir / resta_subdir
    if not resta_dir.is_dir():
        LOGGER.debug("La carpeta %s no contiene el subdirectorio %s", seq_dir, resta_subdir)
        return []
    return sorted(resta_dir.glob(pattern))


def run_binarization(config: BinarizationConfig) -> None:
    """Ejecuta el proceso completo utilizando ``config``."""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level)

    seq_dirs = collect_sequence_dirs(config.root)

    enhancer_kwargs = {
        "sigma_background": config.sigma_background,
        "alpha": config.alpha,
    }
    process_kwargs = {
        "suavizado": config.suavizado,
        "ganancia_tanh": config.ganancia_tanh,
        "mostrar": False,
        "percentil_contornos": config.percentil_contornos,
        "min_dist_picos": config.min_dist_picos,
        "metodo_contorno": config.metodo_contorno,
        "usar_dos_picos": not config.usar_un_solo_pico,
    }

    selected_names = config.images

    for seq_dir in seq_dirs:
        output_dir = ensure_output_dir(config.output_dir, seq_dir)
        for image_path in iter_resta_images(seq_dir, config.resta_subdir, config.image_pattern):
            if not should_process(image_path, selected_names):
                LOGGER.debug("Se omite %s por no estar en la lista solicitada", image_path.name)
                continue

            LOGGER.info("Procesando %s", image_path)
            try:
                binary_full = process_image(image_path, enhancer_kwargs, process_kwargs, config.roi)
            except Exception as exc:  # pragma: no cover - reportamos errores y continuamos
                LOGGER.error("Error procesando %s: %s", image_path, exc)
                continue

            output_path = output_dir / f"{image_path.stem}_binarizada.tiff"
            save_binary_image(output_path, binary_full)
            LOGGER.info("Imagen binarizada guardada en %s", output_path)


if __name__ == "__main__":  # pragma: no cover - punto de entrada del script
    # ------------------------------------------------------------------
    # CONFIGURACIÓN
    # ------------------------------------------------------------------
    CONFIG = BinarizationConfig(
        root=Path("/ruta/a/tu/directorio/base"),
        # output_dir=Path("/ruta/opcional/de/salida"),
        # images=("nombre_imagen_1", "nombre_imagen_2"),
        # roi=(100, 200, 512, 512),  # (x, y, ancho, alto)
        sigma_background=100.0,
        alpha=0.0,
        suavizado=5,
        ganancia_tanh=0.1,
        percentil_contornos=0.0,
        min_dist_picos=30,
        metodo_contorno="sobel",
        usar_un_solo_pico=False,
        log_level="INFO",
    )

    run_binarization(CONFIG)
