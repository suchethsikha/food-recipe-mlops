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
    results = model.predict(img, conf=0.1)
    
    # Extract predicted labels
    predictions = []
    for r in results:
        classes = r.boxes.cls.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()

        for class_id, conf in zip(classes, confidences):
            label = model.names[int(class_id)]
            predictions.append((label, float(conf)))

    return predictions


# app/predictor.py

#import numpy as np
#from ultralytics import YOLO
#from PIL import Image
#import io

#from app.model_loader import model

#def predict_image(file_bytes: bytes):
#    """
#    Runs YOLO inference at 25% confidence. If no detections are found,
#    retries at a lower threshold (e.g. 0.001) to increase recall.
#    Returns a list of (label, confidence) tuples.
#    """
#    # Load image from bytes
#    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

#    # First pass: try high confidence
#    results = model.predict(img, conf=0.25)
#    predictions = _extract_predictions(results)

    # If nothing found, retry at a very low confidence
#    if not predictions:
#        results = model.predict(img, conf=0.001)
#        predictions = _extract_predictions(results)

#    return predictions

#def _extract_predictions(results):
#    """
#    Helper to pull (label, confidence) out of a YOLO Results object list.
#    """
#    preds = []
#    for r in results:
#        classes = r.boxes.cls.cpu().numpy()
#        confidences = r.boxes.conf.cpu().numpy()
#        for class_id, conf in zip(classes, confidences):
#            label = model.names[int(class_id)]
#            preds.append((label, float(conf)))
    # sort by confidence descending
#    preds.sort(key=lambda x: x[1], reverse=True)
#    return preds
