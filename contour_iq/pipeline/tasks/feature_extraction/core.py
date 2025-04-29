import cv2
import numpy as np
from typing import Dict
from skimage.morphology import skeletonize
from scipy.fft import fft

def extract_shape_features(contour: np.ndarray, mask_shape: tuple = None) -> Dict[str, float]:
    """
    Extract basic shape descriptors from a single contour.

    Parameters:
    - contour: NumPy array representing the contour.

    Returns:
    - Dictionary of shape features.
    """
    features = {}

    # Area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Shape descriptors
    features["area"] = area
    features["perimeter"] = perimeter
    features["circularity"] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Bounding box and aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    features["aspect_ratio"] = float(w) / h if h > 0 else 0

    # Extent (object area / bounding box area)
    features["extent"] = area / (w * h) if w * h > 0 else 0

    # Solidity (area / convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    features["solidity"] = area / hull_area if hull_area > 0 else 0

    # Hu Moments (7 invariant moments)
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i, val in enumerate(hu_moments):
        features[f"hu_moment_{i+1}"] = float(val)

    # Convexity Defects
    if len(contour) >= 4:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is not None and len(hull_indices) > 3:
            defects = cv2.convexityDefects(contour, hull_indices)
            features["num_defects"] = 0 if defects is None else defects.shape[0]

    # Eccentricity (requires at least 5 points to fit an ellipse)
    if len(contour) >= 5:
        try:
            _, (MA, ma), _ = cv2.fitEllipse(contour)
            a = max(MA, ma) / 2
            b = min(MA, ma) / 2
            features["eccentricity"] = np.sqrt(1 - (b**2 / a**2)) if MA > 0 else 0
        except:
            features["eccentricity"] = 0

    # Corner count via polygon approximation
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    features["num_corners"] = len(approx)

    # Fourier Descriptor (first harmonic magnitude)
    contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]
    fd = fft(contour_complex)
    if len(fd) > 1:
        features["fourier_1_mag"] = np.abs(fd[1])

    # Skeleton Features (if mask shape provided)
    if mask_shape is not None:
        mask = np.zeros(mask_shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        skeleton = skeletonize(mask).astype(np.uint8)
        skeleton_length = np.count_nonzero(skeleton)
        features["skeleton_length"] = skeleton_length


    return features