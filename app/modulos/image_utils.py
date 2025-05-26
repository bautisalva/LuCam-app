import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from PyQt5.QtGui import QImage, QPixmap
from skimage.color import rgb2gray

def blur_uint16(image, sigma):
    """
    Aplica un desenfoque gaussiano a una imagen uint16.

    Parameters:
        image (np.ndarray): imagen en escala de grises uint16.
        sigma (float): desvío estándar del kernel gaussiano.

    Returns:
        np.ndarray: imagen desenfocada (uint16).
    """
    return gaussian_filter(image, sigma=sigma, mode='reflect').astype(np.uint16)

def to_8bit_for_preview(image_16bit):
    """
    Convierte una imagen de 16 bits a 8 bits para visualización.

    Parameters:
        image_16bit (np.ndarray): imagen uint16.

    Returns:
        np.ndarray: imagen uint8 escalada entre 0 y 255.
    """
    min_val = np.min(image_16bit)
    max_val = np.max(image_16bit)
    if max_val == min_val:
        return np.zeros_like(image_16bit, dtype=np.uint8)
    scaled = ((image_16bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return scaled

def resize_image(image, shape):
    """
    Redimensiona una imagen manteniendo el rango de valores original.

    Parameters:
        image (np.ndarray): imagen original.
        shape (tuple): nueva forma (alto, ancho).

    Returns:
        np.ndarray: imagen redimensionada.
    """
    return resize(image, shape, preserve_range=True).astype(image.dtype)

def prepare_for_qpixmap(image_16bit, target_size=(960, 720)):
    """
    Prepara una imagen uint16 para mostrarse como QPixmap en PyQt.

    Parameters:
        image_16bit (np.ndarray): imagen original en uint16.
        target_size (tuple): tamaño deseado (ancho, alto).

    Returns:
        QPixmap: imagen convertida para QLabel.
    """
    if len(image_16bit.shape) == 3:
        image_16bit = (rgb2gray(image_16bit) * 65535).astype(np.uint16)

    image_8bit = to_8bit_for_preview(image_16bit)
    resized = resize_image(image_8bit, target_size)

    height, width = resized.shape
    bytes_per_line = width
    qimage = QImage(
        resized.data, width, height, bytes_per_line, QImage.Format_Grayscale8
    )
    return QPixmap.fromImage(qimage)
