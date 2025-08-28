"""
main.py
Script demostrativo para probar las funciones de la librería cvtools.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from cvtools import camera, color, filters


def demo_camera():
    print("=== DEMO CAMERA ===")
    pts = np.array([[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]])
    k1, k2 = 0.01, -0.001
    'apply_radial_distortion'
    distorted = camera.apply_radial_distortion(pts, k1, k2)
    print("Puntos originales:", pts)
    print("Puntos distorsionados:", distorted)

    'project_points_pinhole'
    pts3d = np.array([[1.0, 1.0, 2.0],
                      [2.0, 1.0, 4.0],
                      [0.5, 0.5, 1.0]])
    for f in [200, 500, 1000]:
        proj = camera.project_points_pinhole(pts3d, f)
        print(f"Proyección con f={f}: {proj}")



def demo_color():
    print("\n=== DEMO COLOR ===")
    img_bgr = cv2.imread("data/ejemplo1.jpeg")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    'Conversión RGB→HSV y RGB→LAB'
    hsv = color.rgb_to_hsv(img_rgb)
    lab = color.rgb_to_lab(img_rgb)
    print("Conversión RGB→HSV y RGB→LAB realizada.")

        # Mostrar una parte de cada espacio de color
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img_rgb); plt.title("RGB")
    plt.subplot(1,3,2); plt.imshow(hsv); plt.title("HSV")
    plt.subplot(1,3,3); plt.imshow(lab); plt.title("LAB")
    plt.show()

    'color_histogram'
    
    _ = color.color_histogram(img_rgb, bins=64, show=True)
    
    'quantize_image'
    img_q = color.quantize_image(img_rgb, 64)
    plt.figure(); plt.imshow(img_q); plt.title("Imagen cuantizada (64 colores)")
    plt.show()

    'reduce_image_size_by_color'
    img_r, size_kb = color.reduce_image_size_by_color(img_rgb, 32)
    print(f"Imagen reducida a 32 colores → {size_kb:.2f} KB")
    plt.figure(); plt.imshow(img_r); plt.title("Imagen reducida (32 colores)")
    plt.show()


def demo_filters():
    print("\n=== DEMO FILTERS ===")
    img_gray = cv2.imread("data/ejemplo1.jpeg", cv2.IMREAD_GRAYSCALE)

    'convolve2d'
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])  # sharpen
    conv = filters.convolve2d(img_gray, kernel)
    plt.figure(); plt.imshow(conv, cmap="gray"); plt.title("Convolución genérica")
    plt.show()

    "sobel_x y sobel_y"
    sobx = filters.sobel_x(img_gray)
    soby = filters.sobel_y(img_gray)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(sobx, cmap="gray"); plt.title("Sobel X")
    plt.subplot(1,2,2); plt.imshow(soby, cmap="gray"); plt.title("Sobel Y")
    plt.show()

    'canny'
    edges = filters.canny(img_gray, 100, 200)
    plt.figure(); plt.imshow(edges, cmap="gray"); plt.title("Canny")
    plt.show()

    'Laplaciano'
    lap = filters.laplacian(img_gray)
    plt.figure(); plt.imshow(lap, cmap="gray"); plt.title("Laplaciano")
    plt.show()


if __name__ == "__main__":
    demo_camera()
    demo_color()
    demo_filters()
    print("\nDemostración completa ✅")
