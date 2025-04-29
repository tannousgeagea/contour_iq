import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from ultralytics import YOLO
from PIL import Image
from pipeline.main import run_contour_pipeline


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

def main(image_path, model, output_dir=".", debug=False):
    image = cv2.imread(image_path)

    # Run the model
    results = model(image)
    segments = yolo_segmentation_to_masks(results, image.shape)
    output = run_contour_pipeline(image, segments)
    os.makedirs(f"{output_dir}", exist_ok=True)
    
    cv_image = cv2.cvtColor(output["annotated_image"], cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    pil_image.save(f"{output_dir}/{os.path.basename(image_path)}", format="JPEG", quality=60, optimize=True)
    # cv2.imwrite(f"{output_dir}/{os.path.basename(image_path)}", output["annotated_image"])

    if debug:
        os.makedirs(f"{output_dir}/{os.path.basename(image_path).split('.jpg')[0]}", exist_ok=True)
        pbar = tqdm(output["object_images"], ncols=125)
        for i, obj_img in enumerate(pbar):
            cv_image = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image) 
            pil_image.save(f"{output_dir}/{os.path.basename(image_path).split('.jpg')[0]}/object_{i+1}_features.png", format='JPEG', quality=60, optimize=True)
            # cv2.imwrite(f"{output_dir}/{os.path.basename(image_path).split('.jpg')[0]}/object_{i+1}_features.png", obj_img)

if __name__ == "__main__":
    model = YOLO('/media/base.segment.pt')

    images = glob("/media/AGR/images_g1/processed/*.jpg")
    pbar = tqdm(images, ncols=125)
    for image in pbar:
        main(
            image_path=image, 
            model=model,
            output_dir="/media/debug/AGR",
            debug=False
        )
