import cv2
import numpy as np

def contour_perimeter(contour: np.ndarray):
    return cv2.arcLength(contour, True)