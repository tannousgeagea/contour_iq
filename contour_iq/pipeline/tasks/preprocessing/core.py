# contour_analysis_pipeline/pipeline/preprocessing.py

import os
import cv2
import numpy as np
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor

def polygon_to_mask(polygon: List[tuple], image_shape: tuple) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=1)
    return mask

# def preprocess_segmentation(image: np.ndarray, segments: List[Union[np.ndarray, List[tuple]]]) -> List[np.ndarray]:
#     """
#     Normalize input segments to a list of cleaned binary masks.
#     """
#     masks = []
#     for seg in segments:
#         if isinstance(seg, list):  # Polygon format
#             mask = polygon_to_mask(seg, image.shape)
#         else:  # Assume it's already a binary mask
#             mask = seg
#         # Optional: apply morphological ops to clean mask
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         masks.append(mask)
#     return masks

def clean_mask(mask: np.ndarray, apply_morphology: bool = True) -> np.ndarray:
    if not apply_morphology:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def process_segment(seg: Union[np.ndarray, List[tuple]], image_shape: tuple, apply_morphology: bool = True) -> np.ndarray:
    if isinstance(seg, list):
        mask = polygon_to_mask(seg, image_shape)
    else:
        mask = seg
    return clean_mask(mask, apply_morphology=apply_morphology)

def preprocess_segmentation(
    image: np.ndarray,
    segments: List[Union[np.ndarray, List[tuple]]],
    apply_morphology: bool = False,
    max_workers:int = None,
) -> List[np.ndarray]:
    """
    Normalize input segments to a list of cleaned binary masks.
    Utilizes parallel processing for performance.

    Parameters:
    - image: Input image used to determine shape
    - segments: List of binary masks or polygons
    - apply_morphology: If True, apply morphological cleaning

    Returns:
    - List of processed binary masks
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 4 

    print(f"[Preprocessing] Using {max_workers} threads for {len(segments)} segments.")


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda s: process_segment(s, image.shape, apply_morphology), segments
        )
    return list(results)