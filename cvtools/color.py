import numpy as np
import matplotlib.pyplot as plt
import cv2

def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen RGB a HSV usando OpenCV.

    Parameters
    ----------
    img : np.ndarray
        Imagen en formato RGB (uint8 o float).

    Returns
    -------
    hsv : np.ndarray
        Imagen en espacio HSV.
    """
    if img is None:
        raise ValueError("La imagen no puede ser None")
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen RGB a CIELAB usando OpenCV.

    Parameters
    ----------
    img : np.ndarray
        Imagen en formato RGB (uint8 o float).

    Returns
    -------
    lab : np.ndarray
        Imagen en espacio CIELAB.
    """
    if img is None:
        raise ValueError("La imagen no puede ser None")
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


def color_histogram(img: np.ndarray, bins: int = 256, show: bool = True):
    """
    Calcula y grafica el histograma de colores de una imagen RGB.

    Parameters
    ----------
    img : np.ndarray
        Imagen en formato RGB (uint8).
    bins : int, optional
        Número de bins para el histograma (default=256).
    show : bool, optional
        Si es True, grafica el histograma.

    Returns
    -------
    histograms : dict
        Diccionario con los histogramas de cada canal {"r": hist_r, "g": hist_g, "b": hist_b}.
    """
    if img is None:
        raise ValueError("La imagen no puede ser None")

    # Separar canales
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    # Calcular histogramas
    hist_r, _ = np.histogram(r, bins=bins, range=(0, 256))
    hist_g, _ = np.histogram(g, bins=bins, range=(0, 256))
    hist_b, _ = np.histogram(b, bins=bins, range=(0, 256))

    if show:
        plt.figure(figsize=(10,5))
        plt.plot(hist_r, color="red", label="Rojo")
        plt.plot(hist_g, color="green", label="Verde")
        plt.plot(hist_b, color="blue", label="Azul")
        plt.title("Histograma de colores")
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.show()

    return {"r": hist_r, "g": hist_g, "b": hist_b}


def quantize_image(img: np.ndarray, n_colors: int) -> np.ndarray:
    """
    Cuantiza una imagen RGB reduciendo el número de colores posibles.

    Parameters
    ----------
    img : np.ndarray
        Imagen en formato RGB (uint8).
    n_colors : int
        Número total de colores deseados (ej: 256, 64, 16).
        Debe ser potencia de 2 (ej: 2, 4, 16, 64, 256).

    Returns
    -------
    quantized_img : np.ndarray
        Imagen cuantizada con n_colors posibles.
    """
    if img is None:
        raise ValueError("La imagen no puede ser None")

    # Calcular niveles por canal
    levels = int(round(n_colors ** (1/3)))  # raíz cúbica para repartir entre R,G,B
    if levels < 2:
        levels = 2  # mínimo 2 niveles

    # Crear intervalos de cuantización
    bins = np.linspace(0, 256, levels+1, endpoint=True)

    # Cuantizar cada canal
    img_q = np.zeros_like(img)
    for c in range(3):  # R,G,B
        channel = img[:,:,c]
        inds = np.digitize(channel, bins) - 1
        inds = np.clip(inds, 0, levels-1)
        # Asignar valor promedio de cada bin
        img_q[:,:,c] = (bins[inds] + bins[inds+1]) / 2

    return img_q.astype(np.uint8)

def reduce_image_size_by_color(img: np.ndarray, n_colors: int) -> tuple[np.ndarray, float]:
    """
    Reduce la cantidad de colores de la imagen y retorna la imagen cuantizada + tamaño en KB.

    Parameters
    ----------
    img : np.ndarray
        Imagen en formato RGB (uint8).
    n_colors : int
        Número de colores a mantener (ej: 256, 64, 16).

    Returns
    -------
    quantized_img : np.ndarray
        Imagen cuantizada.
    size_kb : float
        Tamaño de la imagen cuantizada en KB.
    """
    if img is None:
        raise ValueError("La imagen no puede ser None")

    # === Paso 1: Cuantizar colores (versión simple) ===
    levels = int(round(n_colors ** (1/3)))
    if levels < 2:
        levels = 2

    bins = np.linspace(0, 256, levels+1, endpoint=True)
    img_q = np.zeros_like(img)

    for c in range(3):  # R, G, B
        channel = img[:,:,c]
        inds = np.digitize(channel, bins) - 1
        inds = np.clip(inds, 0, levels-1)
        img_q[:,:,c] = (bins[inds] + bins[inds+1]) / 2

    img_q = img_q.astype(np.uint8)

    # === Paso 2: Guardar en memoria para medir tamaño ===
    success, encoded_img = cv2.imencode(".png", cv2.cvtColor(img_q, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Error al codificar la imagen cuantizada")

    size_kb = len(encoded_img.tobytes()) / 1024.0  # tamaño en KB

    return img_q, size_kb


