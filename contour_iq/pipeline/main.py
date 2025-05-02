
import cv2
from PIL import Image 
from typing import List, Dict, Union
import numpy as np
from common_utils.time_tracker.core import KeepTrackOfTime
from pipeline.tasks.preprocessing import preprocess_segmentation
from pipeline.tasks.contour_extraction import extract_all_contours
from pipeline.tasks.feature_extraction import extract_shape_features
from pipeline.tasks.analysis import analyze_contour
from pipeline.tasks.annotation import annotate_image

keep_track_of_time = KeepTrackOfTime()

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
        segments: List[Union[np.ndarray, List[tuple]]],
        render_individual:bool=False,
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

    keep_track_of_time.start(task="run_pipeline")

    keep_track_of_time.start(task='preprocessing')
    masks = preprocess_segmentation(image, segments)
    keep_track_of_time.end(task='preprocessing')

    keep_track_of_time.start(task='extract_contour')
    all_contours = extract_all_contours(masks)
    keep_track_of_time.end(task='extract_contour')

    all_features = []
    all_attributes = []
    flat_contours = []

    keep_track_of_time.start(task='extract_feature')
    for i, contours in enumerate(all_contours):
        for contour in contours:
            keep_track_of_time.start(task='extract_feature_per_contour')
            features = extract_shape_features(contour, mask_shape=image.shape[:2])
            attributes = analyze_contour(features)

            all_features.append({**features, **attributes})
            all_attributes.append(attributes)
            flat_contours.append(contour)
            keep_track_of_time.end(task='extract_feature_per_contour')
            keep_track_of_time.log(task='extract_feature_per_contour', prefix="Per-Contour Feature Extraction")
    keep_track_of_time.end(task='extract_feature')

    keep_track_of_time.log(task='preprocessing', prefix="Preprocissing Time")
    keep_track_of_time.log(task='extract_contour', prefix="Extract contour Time")
    keep_track_of_time.log(task='extract_feature', prefix="Feature Extraction")

    keep_track_of_time.end(task="run_pipeline")
    keep_track_of_time.log(task="run_pipeline", prefix="Total Execution Time")
    
    annotated = annotate_image(image, flat_contours, all_attributes)
    individual_images = render_individual_features(image, flat_contours, all_features) if render_individual else []

    return {
        "annotated_image": annotated,
        "results": all_features,
        "object_images": individual_images
    }

