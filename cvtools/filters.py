import numpy as np
import cv2

def convolve2d(img: np.ndarray, kernel: np.ndarray, padding: str = "same") -> np.ndarray:
    """
    Aplica convolución 2D a una imagen en escala de grises.

    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada (2D, escala de grises).
    kernel : np.ndarray
        Kernel de convolución (2D).
    padding : str, optional
        "same" → mantiene tamaño de la imagen
        "valid" → sin padding, reduce el tamaño

    Returns
    -------
    output : np.ndarray
        Imagen resultante de la convolución.
    """


    # Dimensiones
    kh, kw = kernel.shape
    ih, iw = img.shape

    # Voltear el kernel (definición de convolución)
    kernel = np.flipud(np.fliplr(kernel))

    # Padding
    if padding == "same":
        pad_h, pad_w = kh // 2, kw // 2
        img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    elif padding == "valid":
        img_padded = img
    else:
        raise ValueError("padding debe ser 'same' o 'valid'")

    # Salida
    oh = ih if padding == "same" else ih - kh + 1
    ow = iw if padding == "same" else iw - kw + 1
    output = np.zeros((oh, ow), dtype=float)

    # Convolución manual
    for y in range(oh):
        for x in range(ow):
            region = img_padded[y:y+kh, x:x+kw]
            output[y, x] = np.sum(region * kernel)

    return output

def sobel_x(img: np.ndarray) -> np.ndarray:
    """
    Aplica el filtro Sobel en X (detecta bordes verticales).

    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises.

    Returns
    -------
    np.ndarray
        Imagen filtrada.
    """
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    return convolve2d(img, kernel, padding="same")


def sobel_y(img: np.ndarray) -> np.ndarray:
    """
    Aplica el filtro Sobel en Y (detecta bordes horizontales).

    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises.

    Returns
    -------
    np.ndarray
        Imagen filtrada.
    """
    kernel = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]])
    return convolve2d(img, kernel, padding="same")

def canny(img: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """
    Aplica el detector de bordes de Canny.

    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises.
    low : int
        Umbral inferior para histéresis.
    high : int
        Umbral superior para histéresis.

    Returns
    -------
    edges : np.ndarray
        Imagen binaria con los bordes detectados.
    """
    if img is None:
        raise ValueError("La imagen no puede ser None")
    if img.ndim != 2:
        raise ValueError("Canny requiere una imagen en escala de grises (2D)")

    # Aplicar detector de Canny
    edges = cv2.Canny(img, low, high)

    return edges

def laplacian(img: np.ndarray) -> np.ndarray:
    """
    Aplica el filtro Laplaciano para resaltar bordes.

    Parameters
    ----------
    img : np.ndarray
        Imagen en escala de grises.

    Returns
    -------
    laplace_img : np.ndarray
        Imagen resultante después de aplicar el Laplaciano.

    Notes
    -----
    El filtro Laplaciano resalta:
    - Bordes (cambios bruscos de intensidad).
    - Zonas de alta frecuencia (también puede amplificar el ruido).
    No distingue orientación del borde, solo su presencia.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    return convolve2d(img, kernel, padding="same")
