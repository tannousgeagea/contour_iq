
import cv2
import time
import json
import numpy as np
from fastapi import HTTPException
from fastapi import FastAPI, File, UploadFile, Body, Depends, Form, Query
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi import Request, Response
from typing import Callable, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import io
from ultralytics import YOLO
from pipeline.main import run_contour_pipeline

model = YOLO('/media/amk.front.segmentation.v1.pt')

# Define the Pydantic models for the response
class Label(BaseModel):
    id: str
    x: float
    y: float
    attributes: List[str]

class Features(BaseModel):
    area: float
    circularity: float = None
    eccentricity: float = None
    solidity: float = None
    extent: float = None
    aspect_ratio: float = None
    skeleton_length: float = None
    num_corners: int = None
    num_defects: int = None
    # area_skeleton_ratio: float

class Contour(BaseModel):
    id: str
    points: List[dict]
    color: str
    labels: List[Label]
    features: Features

class AnalyzedImage(BaseModel):
    originalSrc: str
    width: int
    height: int
    contours: List[Contour]

class Threshold(BaseModel):
    name: str
    value: float

class Attribute(BaseModel):
    name: str
    value: bool


class TimedRoute(APIRoute):
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request):
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            return response

        return custom_route_handler

router = APIRouter(route_class=TimedRoute)


def yolo_segmentation_to_masks(results, image_shape):
    segments = []
    if results[0].masks is None:
        return segments
    for mask in results[0].masks.data:
        binary_mask = mask.cpu().numpy().astype(np.uint8) * 255
        resized_mask = cv2.resize(binary_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        segments.append((resized_mask > 127).astype(np.uint8))
    return segments

# Function to simulate contour analysis (mock implementation)
def analyze_image_with_thresholds(image_bytes: bytes, thresholds: List[Threshold], attributes: List[Attribute]) -> AnalyzedImage:
    cv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = cv_image.shape

    contours = []
    results = model(cv_image)
    masks = yolo_segmentation_to_masks(results, cv_image.shape)
    output = run_contour_pipeline(cv_image, masks, render_individual=False)

    for i, obj in enumerate(output['contours']):
        print(output["attributes"][i])
        M = cv2.moments(obj)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        color = "rgba(36, 99, 235, 0.4)" if output["attributes"][i].get("manmade") else "rgba(255, 0, 0, 0.4)"
        color = "rgba(255, 165, 0, 0.4)" if output["attributes"][i].get("long") else color
        color = "rgba(255, 0, 165, 0.4)" if output["attributes"][i].get("rigid") else color

        contours.append({
            "id": str(i),
            "points": [
                {
                    "x": int(x),
                    "y": int(y)
                } for x, y in obj.squeeze()
            ],
            "color": color,
            "labels": [{"id": f"{i}-1", "x": cx, "y": cy, "attributes": [attr for attr, v in output['attributes'][i].items() if v]}],
            "features": output["results"][i],
        })

    # Return the analyzed image with contours
    return AnalyzedImage(
        originalSrc="mocked_image_url",  # URL of the uploaded image
        width=width,
        height=height,
        contours=contours
    )


# FastAPI endpoint to handle image and thresholds
@router.post("/analyze_image")
async def analyze_image(request: Request, image: UploadFile = File(...)):
    """
    Analyze the uploaded image with the given thresholds and attributes.
    Returns the contours and features of the detected objects.
    """
    form_data = await request.form()
    thresholds = form_data.get('thresholds')
    attributes = form_data.get("attributes")

    image_bytes = await image.read()
    result = analyze_image_with_thresholds(image_bytes, thresholds=thresholds, attributes=attributes)
    return JSONResponse(content=result.model_dump())
