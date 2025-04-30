import cv2
import numpy as np

def contour_extent(contour_area:float, w:float, h:float):
    return contour_area / (w * h) if w * h > 0 else 0