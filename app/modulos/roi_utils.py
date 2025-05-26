
def validar_roi(x, y, w, h, img_shape):
    """
    Verifica si un ROI es válido dentro de las dimensiones de la imagen.

    Parameters:
        x, y (int): coordenadas superiores izquierdas del ROI
        w, h (int): ancho y alto del ROI
        img_shape (tuple): (alto, ancho) de la imagen original

    Returns:
        bool: True si el ROI es válido
    """
    alto, ancho = img_shape
    return w >= 16 and h >= 16 and x + w <= ancho and y + h <= alto

def aplicar_roi(imagen, x, y, w, h, habilitado=True):
    """
    Recorta la imagen con el ROI si es válido y está habilitado.

    Returns:
        np.ndarray: imagen recortada o completa si ROI inválido.
    """
    if not habilitado or not validar_roi(x, y, w, h, imagen.shape):
        return imagen.copy()
    return imagen[y:y+h, x:x+w]
