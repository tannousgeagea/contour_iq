import cv2
import numpy as np


def contour_aspect_ratio(w:float, h:float):
    return float(w) / h if h > 0 else 0