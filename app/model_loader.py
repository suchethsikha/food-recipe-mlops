# app/model_loader.py

from ultralytics import YOLO

# Load model once at startup
model = YOLO("best_model/best.pt")  # Update path if needed
