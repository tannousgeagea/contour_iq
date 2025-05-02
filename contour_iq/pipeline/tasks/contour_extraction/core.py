import os
import cv2
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor

def extract_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    Extracts external contours from a binary mask.

    Parameters:
    - mask (np.ndarray): Binary mask of a single object.

    Returns:
    - List of contours (each contour is a NumPy array of points).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_all_contours(masks: List[np.ndarray], max_workers:int=None) -> List[List[np.ndarray]]:
    """
    Extracts contours for a list of binary masks using parallel processing.

    Parameters:
    - masks: List of binary masks.

    Returns:
    - List of lists of contours (per object).
    """
    return [extract_contours(mask) for mask in masks]