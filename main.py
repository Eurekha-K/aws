from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.staticfiles import StaticFiles
import uuid
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
from mangum import Mangum
 
app = FastAPI()
 
# Allow CORS for your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with the actual origin of your React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
IMAGE_DIR = "stored_images"
recent_image_path = None
model_d = YOLO("best.pt")
app.mount("/static", StaticFiles(directory="static"), name="static")
 
@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    global recent_image_path
    global model_d
 
    original_filename = image.filename
    filename = f"{uuid.uuid4()}.{original_filename.split('.')[-1]}"
    save_path = os.path.join(IMAGE_DIR, filename)
 
    with open(save_path, "wb") as f:
        contents = await image.read()
        f.write(contents)
 
    recent_image_path = save_path
 
    model_c = YOLO("classification.pt")
    results_c = model_c(recent_image_path)
    names_dict = results_c[0].names
    probs = results_c[0].probs.data.tolist()
    classification_result = names_dict[np.argmax(probs)]
    image = cv2.imread(recent_image_path)
    result = model_d(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > 0.4]
    labels = [result.names[class_id] for class_id in detections.class_id]
    class_colors = sv.ColorPalette(colors=[
        sv.Color(244, 250, 161),
        sv.Color(211, 211, 211),
        sv.Color(119, 165, 255),
        sv.Color(246, 167, 85),
        sv.Color(176, 212, 200),
        sv.Color(253, 205, 252)
    ])
    bbox_annotator = sv.BoxAnnotator(color=class_colors)
    annotated_image = bbox_annotator.annotate(scene=image, detections=detections, labels=labels)
    cv2.imwrite('static/damaged_car.png', annotated_image)
 
    detection_plot_url = "/static/damaged_car.png"
    image = cv2.imread(recent_image_path)
    result = model_d(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > 0.6]
    labels = tuple(set([result.names[class_id] for class_id in detections.class_id]))
 
    return JSONResponse(content={
        "message": "Image uploaded successfully",
        "classification_result": classification_result,
        "detection_plot_url": detection_plot_url,
        "labels_with_confidence": labels
    })