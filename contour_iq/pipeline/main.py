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
            cv2.putText(temp, text, (10, y0 + i * 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

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

import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from ultralytics import YOLO


def yolo_segmentation_to_masks(results, image_shape):
    """
    Extract and resize YOLOv8 segmentation masks to match the input image shape.
    """
    segments = []
    for mask in results[0].masks.data:  # Each mask is [h, w] tensor
        binary_mask = mask.cpu().numpy().astype(np.uint8) * 255
        resized_mask = cv2.resize(binary_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        segments.append((resized_mask > 127).astype(np.uint8))  # Ensure binary
    return segments

def main(image_path, model):
    image = cv2.imread(image_path)


    # Run the model
    results = model(image)

    # Convert segmentation masks
    segments = yolo_segmentation_to_masks(results, image.shape)

    # Run ContourIQ pipeline
    output = run_contour_pipeline(image, segments)

    # Save annotated output
    cv2.imwrite("output_yolo_seg.png", output["annotated_image"])
    print("Saved: output_yolo_seg.png")



    os.makedirs(f"/media/debug/{os.path.basename(image_path).split('.jpg')[0]}", exist_ok=True)
    pbar = tqdm(output["object_images"], ncols=125)
    for i, obj_img in enumerate(pbar):
        cv2.imwrite(f"/media/debug/{os.path.basename(image_path).split('.jpg')[0]}/object_{i+1}_features.png", obj_img)

if __name__ == "__main__":
    model = YOLO('/media/amk.front.segmentation.v1.pt')
    images = '/media/SWB/images/1715322859.jpg'

    images = glob("/media/AGR/snapshots_before/*.jpg")
    pbar = tqdm(images, ncols=125)
    for image in pbar:
        main(
            image_path=image, 
            model=model
        )

