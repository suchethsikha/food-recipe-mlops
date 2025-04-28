# app/model_loader.py

from ultralytics import YOLO

# Load model once at startup
model = YOLO("/opt/homebrew/runs/detect/train7/weights/best.pt")  # Update path if needed
