# app/predictor.py

import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

from app.model_loader import model

def predict_image(file_bytes: bytes):
    # Load image from bytes
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Run prediction
    results = model.predict(img, conf=0.25)

    # Extract predicted labels
    predictions = []
    for r in results:
        classes = r.boxes.cls.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()

        for class_id, conf in zip(classes, confidences):
            label = model.names[int(class_id)]
            predictions.append((label, float(conf)))

    return predictions