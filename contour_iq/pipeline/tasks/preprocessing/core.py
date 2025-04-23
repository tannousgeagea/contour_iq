# contour_analysis_pipeline/pipeline/preprocessing.py

import cv2
import numpy as np
from typing import List, Union

def polygon_to_mask(polygon: List[tuple], image_shape: tuple) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=1)
    return mask

def preprocess_segmentation(image: np.ndarray, segments: List[Union[np.ndarray, List[tuple]]]) -> List[np.ndarray]:
    """
    Normalize input segments to a list of cleaned binary masks.
    """
    masks = []
    for seg in segments:
        if isinstance(seg, list):  # Polygon format
            mask = polygon_to_mask(seg, image.shape)
        else:  # Assume it's already a binary mask
            mask = seg
        # Optional: apply morphological ops to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masks.append(mask)
    return masks
