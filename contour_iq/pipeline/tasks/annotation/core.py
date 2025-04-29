import cv2
import numpy as np
from typing import List, Dict


def annotate_image(image: np.ndarray, contours: List[np.ndarray], attributes_list: List[Dict[str, bool]]) -> np.ndarray:
    """
    Draw contours and label each object with all classification attributes.

    Parameters:
    - image: The original image
    - contours: List of contours (1 per object)
    - attributes_list: List of dictionaries with shape attributes for each contour

    Returns:
    - Annotated image with labels drawn.
    """
    annotated = image.copy()
    overlay = image.copy()

    for i, (cnt, attrs) in enumerate(zip(contours, attributes_list)):
        color = (0, 255, 0) if attrs.get("is_man_made") else (0, 0, 255)
        color = (0, 165, 255) if attrs.get("long_object") else color
        color = (165, 0, 255) if attrs.get("rigid_object") else color
        cv2.drawContours(overlay, [cnt], -1, color, thickness=cv2.FILLED)
        cv2.drawContours(annotated, [cnt], -1, color, 2)

        labels = [k.replace('_', ' ').title() for k, v in attrs.items() if v is True]

        if cnt.shape[0] > 0:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                for j, label in enumerate(labels):
                    cv2.putText(
                        annotated,
                        label,
                        (cx, cy + j * 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 255, 255),
                        1
                    )
    annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)

    return annotated
