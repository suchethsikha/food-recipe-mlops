# app/model_loader.py

from ultralytics import YOLO

# Load model once at startup
model = YOLO("app/model.pt")
