import numpy as np

def apply_radial_distortion(points: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """
    Aplica distorsión radial a puntos en coordenadas de imagen normalizadas.

    Parameters
    ----------
    points : (N, 2) np.ndarray
        Puntos (x, y) en coordenadas normalizadas.
    k1, k2 : float
        Parámetros de distorsión radial.

    Returns
    -------
    distorted_points : (N, 2) np.ndarray
        Puntos distorsionados.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("La entrada debe ser un arreglo de forma (N, 2)")

    x, y = points[:, 0], points[:, 1]
    r2 = x**2 + y**2
    factor = 1 + k1 * r2 + k2 * (r2**2)

    x_distorted = x * factor
    y_distorted = y * factor

    return np.stack([x_distorted, y_distorted], axis=1)


def project_points_pinhole(points_3d: np.ndarray, f: float) -> np.ndarray:
    """
    Proyecta puntos 3D con un modelo pinhole simple.

    Parameters
    ----------
    points_3d : (N, 3) np.ndarray
        Puntos 3D en coordenadas de cámara (X, Y, Z) con Z > 0.
    f : float
        Longitud focal.

    Returns
    -------
    projected_points : (N, 2) np.ndarray
        Puntos proyectados en 2D (x, y).
    """
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("La entrada debe ser un arreglo de forma (N, 3)")

    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    if np.any(Z <= 0):
        raise ValueError("Todos los puntos deben tener Z > 0")

    x = f * (X / Z)
    y = f * (Y / Z)

    return np.stack([x, y], axis=1)



