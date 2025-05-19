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


class Threshold(BaseModel):
    name: str
    value: float

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

class ObjectAnalysis(BaseModel):
    id: str
    features: Features
    attributes: List[str]


class ContoursRequest(BaseModel):
    input_shape: List[int]
    contours: List[List[List[int]]]  # List of contours with each contour as a list of points (x, y)
    thresholds: List[Threshold]  # List of thresholds to classify the objects

class ContoursResponse(BaseModel):
    analyzed_objects: List[ObjectAnalysis]

# Function to analyze the contours based on thresholds
def analyze_contours(contours: List[List[List[int]]], input_shape:tuple, thresholds: List[Threshold]) -> List[ObjectAnalysis]:
    analyzed_objects = []

    cv_image = np.zeros(shape=input_shape, dtype=np.uint8)
    output = run_contour_pipeline(cv_image, segments=contours, render_individual=False)
    
    for i, obj in enumerate(output['contours']):
        analyzed_objects.append(
            ObjectAnalysis(
                id=str(i),
                features=output["results"][i],
                attributes=[attr for attr, v in output['attributes'][i].items() if v]
            )
        )

    return analyzed_objects

@router.api_route("/analyze_contours", methods=["POST"], response_model=ContoursResponse)
async def analyze_contours_api(request: ContoursRequest):
    """
    Receives a list of contours and thresholds, analyzes the contours, and returns the features and attributes.
    """
    try:
        analyzed_objects = analyze_contours(request.contours, request.input_shape, request.thresholds)
        return ContoursResponse(analyzed_objects=analyzed_objects)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))