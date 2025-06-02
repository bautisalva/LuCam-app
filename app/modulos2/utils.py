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



import datetime

def log_message(message, log_file=None, consoles=None):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"

    # Mostrar en terminal
    print(full_message)

    # Mostrar en consolas de las tabs
    if consoles:
        for console in consoles:
            if console:
                console.appendPlainText(full_message)

    # Guardar en archivo
    if log_file:
        try:
            log_file.write(full_message + "\n")
            log_file.flush()
        except Exception as e:
            print(f"[ERROR] Falló escritura en log.txt: {e}")
