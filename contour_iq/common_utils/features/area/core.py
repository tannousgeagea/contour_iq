import cv2
import numpy as np

def contour_area(contour: np.ndarray):
    return cv2.contourArea(contour=contour)