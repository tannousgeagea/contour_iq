import cv2
import numpy as np
from typing import Dict
from skimage.morphology import skeletonize
from scipy.fft import fft
from common_utils.features import (
    contour_area,
    contour_perimeter,
    contour_circularity,
    contour_aspect_ratio,
    contour_extent,
)

from common_utils.time_tracker.core import KeepTrackOfTime
keep_track_of_time = KeepTrackOfTime()

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
    keep_track_of_time.start(task="area")
    area = contour_area(contour)
    keep_track_of_time.end(task="area")
    keep_track_of_time.log(task="area", prefix="AREA")

    keep_track_of_time.start(task="perimeter")
    perimeter = contour_perimeter(contour)
    keep_track_of_time.end(task="perimeter")
    keep_track_of_time.log(task="perimeter", prefix="Perimeter")

    x, y, w, h = cv2.boundingRect(contour)
    
    # Shape descriptors
    features["area"] = area
    features["perimeter"] = perimeter
    
    keep_track_of_time.start(task="circularity")
    features["circularity"] = contour_circularity(contour, perimeter, area)
    keep_track_of_time.end(task="circularity")
    keep_track_of_time.log(task="circularity", prefix="Circularity")

    keep_track_of_time.start(task="aspect_ration")
    features["aspect_ratio"] = contour_aspect_ratio(w=w, h=h)
    keep_track_of_time.end(task="aspect_ration")
    keep_track_of_time.log(task="aspect_ration", prefix="Aspect Ration")

    keep_track_of_time.start(task="extent")
    features["extent"] = contour_extent(contour_area=area, w=w, h=h)
    keep_track_of_time.end(task="extent")
    keep_track_of_time.log(task="extent", prefix="Extent")

    # Solidity (area / convex hull area)
    keep_track_of_time.start('solidity')
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    features["solidity"] = area / hull_area if hull_area > 0 else 0
    keep_track_of_time.end(task="solidity")
    keep_track_of_time.log(task="solidity", prefix="Solidity")

    # Hu Moments (7 invariant moments)
    keep_track_of_time.start('hu_moment')
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i, val in enumerate(hu_moments):
        features[f"hu_moment_{i+1}"] = float(val)
    keep_track_of_time.end(task="hu_moment")
    keep_track_of_time.log(task="hu_moment", prefix="HU Moment")

    # Convexity Defects
    keep_track_of_time.start('defect')
    if len(contour) >= 4:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is not None and len(hull_indices) > 3:
            defects = cv2.convexityDefects(contour, hull_indices)
            features["num_defects"] = 0 if defects is None else defects.shape[0]
    keep_track_of_time.end(task="defect")
    keep_track_of_time.log(task="defect", prefix="Defect")

    # Eccentricity (requires at least 5 points to fit an ellipse)
    keep_track_of_time.start('eccentricity')
    if len(contour) >= 5:
        try:
            _, (MA, ma), _ = cv2.fitEllipse(contour)
            a = max(MA, ma) / 2
            b = min(MA, ma) / 2
            features["eccentricity"] = np.sqrt(1 - (b**2 / a**2)) if MA > 0 else 0
        except:
            features["eccentricity"] = 0
    keep_track_of_time.end(task="eccentricity")
    keep_track_of_time.log(task="eccentricity", prefix="Eccentricity")

    # Corner count via polygon approximation
    keep_track_of_time.start('num_corners')
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    features["num_corners"] = len(approx)
    keep_track_of_time.end(task="num_corners")
    keep_track_of_time.log(task="num_corners", prefix="Num corners")

    # Fourier Descriptor (first harmonic magnitude)
    keep_track_of_time.start('fourier_mag')
    contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]
    fd = fft(contour_complex)
    if len(fd) > 1:
        features["fourier_1_mag"] = np.abs(fd[1])
    keep_track_of_time.end(task="fourier_mag")
    keep_track_of_time.log(task="fourier_mag", prefix="Fourier Mag")

    # Skeleton Features (if mask shape provided)
    keep_track_of_time.start('skeleton_length')   
    if mask_shape is not None:
        mask = np.zeros(mask_shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        scale = 0.15
        small_mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        small_skeleton = skeletonize(small_mask).astype(np.uint8)
        skeleton_length = int(np.count_nonzero(small_skeleton) / scale)
        features["skeleton_length"] = skeleton_length
    keep_track_of_time.end(task="skeleton_length")
    keep_track_of_time.log(task="skeleton_length", prefix="Skeleton Length")


    return features