import numpy as np
from scipy.ndimage import gaussian_filter

def blur_uint16(image, sigma):
    """
    Applies Gaussian blur to a 16-bit image using scipy.ndimage.
    
    Parameters:
        image (np.ndarray): uint16 image.
        sigma (float): standard deviation of Gaussian kernel.
    
    Returns:
        np.ndarray: blurred image, dtype uint16.
    """
    return gaussian_filter(image, sigma=sigma, mode='reflect').astype(np.uint16)

def to_8bit_for_preview(image_16bit):
    """
    Escala una imagen uint16 a uint8 para visualización,
    mapeando el rango [min, max] a [0, 255].
    """
    min_val = np.min(image_16bit)
    max_val = np.max(image_16bit)
    if max_val == min_val:
        return np.zeros_like(image_16bit, dtype=np.uint8)
    scaled = ((image_16bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return scaled

