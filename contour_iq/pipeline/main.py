
import cv2
from PIL import Image 
from typing import List, Dict, Union
import numpy as np

from pipeline.tasks.preprocessing import preprocess_segmentation
from pipeline.tasks.contour_extraction import extract_all_contours
from pipeline.tasks.feature_extraction import extract_shape_features
from pipeline.tasks.analysis import analyze_contour
from pipeline.tasks.annotation import annotate_image

def render_individual_features(image: np.ndarray, contours: List[np.ndarray], feature_list: List[Dict[str, float]]) -> List[np.ndarray]:
    """
    Create one image per object with all computed features rendered.

    Parameters:
    - image: Original image
    - contours: List of contours
    - feature_list: Corresponding features for each contour

    Returns:
    - List of annotated images (one per object)
    """
    outputs = []
    for contour, features in zip(contours, feature_list):
        temp = image.copy()
        cv2.drawContours(temp, [contour], -1, (255, 255, 255), 2)

        y0 = 20
        for i, (k, v) in enumerate(features.items()):
            text = f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
            color = (0, 0, 255) if v == True else (0, 255, 0)
            cv2.putText(temp, text, (10, y0 + i * 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        outputs.append(temp)

    return outputs

def run_contour_pipeline(
        image: np.ndarray, 
        segments: List[Union[np.ndarray, List[tuple]]]
        ) -> Dict[str, Union[np.ndarray, List[Dict[str, Union[float, bool]]]]]:
    """
    Full pipeline to analyze object contours from a segmented image.

    Parameters:
    - image: Input image
    - segments: List of binary masks or polygons representing segmented objects

    Returns:
    - Dictionary with:
        'annotated_image': Annotated image with overlays
        'results': List of feature + attribute dicts for each object
    """
    masks = preprocess_segmentation(image, segments)
    all_contours = extract_all_contours(masks)

    all_features = []
    all_attributes = []
    flat_contours = []

    for i, contours in enumerate(all_contours):
        for contour in contours:
            features = extract_shape_features(contour, mask_shape=image.shape[:2])
            attributes = analyze_contour(features)

            all_features.append({**features, **attributes})
            all_attributes.append(attributes)
            flat_contours.append(contour)

    annotated = annotate_image(image, flat_contours, all_attributes)
    individual_images = render_individual_features(image, flat_contours, all_features)

    return {
        "annotated_image": annotated,
        "results": all_features,
        "object_images": individual_images
    }

