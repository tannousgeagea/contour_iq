import cv2
import numpy as np

def contour_circularity(contour: np.ndarray, perimeter:float, area:float):
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0