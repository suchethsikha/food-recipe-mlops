import io
import os
import tempfile
from inference_sdk import InferenceHTTPClient

# Initialize the Roboflow Inference HTTP Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="wFYncWCCTrusFNQRzDpe"
)

MODEL_ID = "food-ingredients-detection-6ce7j/1"

def predict_image(file_bytes: bytes):
    try:
        # Save the uploaded image bytes directly to a temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        # Perform inference
        results = CLIENT.infer(temp_path, model_id=MODEL_ID)

        # Extract predictions
        predictions = []
        if results and 'predictions' in results:
            for pred in results['predictions']:
                label = pred['class']
                confidence = pred['confidence']
                predictions.append((label, float(confidence)))

        return predictions

    except Exception as e:
        print(f"Error during inference: {e}")
        return []

    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

# # app/predictor.py

# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# import io

# from app.model_loader import model

# def predict_image(file_bytes: bytes):
#     # Load image from bytes
#     img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
#     # Run prediction
#     results = model.predict(img, conf=0.25)
    
#     # Extract predicted labels
#     predictions = []
#     print(results)
#     for r in results:
#         classes = r.boxes.cls.cpu().numpy()
#         confidences = r.boxes.conf.cpu().numpy()

#         for class_id, conf in zip(classes, confidences):
#             label = model.names[int(class_id)]
#             predictions.append((label, float(conf)))

#     return predictions
