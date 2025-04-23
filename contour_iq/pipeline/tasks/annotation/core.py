import cv2
import numpy as np
from typing import List, Dict

def annotate_image(image: np.ndarray, contours: List[np.ndarray], attributes_list: List[Dict[str, bool]]) -> np.ndarray:
    """
    Draw contours and label each object with its classification results.

    Parameters:
    - image: The original image
    - contours: List of contours (1 per object)
    - attributes_list: List of dictionaries with shape attributes for each contour

    Returns:
    - Annotated image with labels drawn.
    """
    annotated = image.copy()

    for i, (cnt, attrs) in enumerate(zip(contours, attributes_list)):
        color = (0, 255, 0) if attrs.get("is_man_made") else (0, 0, 255)
        cv2.drawContours(annotated, [cnt], -1, color, 2)

        label = []
        if attrs.get("is_man_made"):
            label.append("Man-made")
        if attrs.get("fracture_detected"):
            label.append("Fractured")
        text = ", ".join(label)

        if cnt.shape[0] > 0:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(annotated, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return annotated
