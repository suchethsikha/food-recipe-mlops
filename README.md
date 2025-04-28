# 🥘 Food Ingredient Detector and Recipe Recommender
This project uses a **YOLOv8 object detection model** trained on food/ingredient images to:

- Detect ingredients from a **photo** uploaded by the user.
- (Future extension) Recommend recipes based on detected ingredients using an LLM.

Built with:
- **FastAPI** backend.
- **Simple HTML frontend**.
- **MLflow** for experiment tracking.
- **Local YOLOv8 training**.

---

## 📸 Quick Demo

1. User uploads a picture (e.g., of a salad or vegetables).
2. Server detects ingredients like **"Tomato", "Lettuce", "Onion"**.
3. *(Coming soon)* Server recommends recipes based on detected items.

---

## 📂 Project Structure

```
food-recipe-mlops/
├── app/                     # FastAPI App
│   ├── main.py              # FastAPI routes
│   ├── model_loader.py      # Load trained YOLO model
│   ├── predictor.py         # Run prediction on uploaded image
│   ├── templates/           # Frontend (HTML)
│   │   └── upload.html
│   └── static/              # (Optional) CSS/JS for styling
├── training/                # Training code
│   ├── train.py             # Train YOLO model, log to MLflow
│   |
├── requirements.txt         # All Python dependencies
├── README.md                # This file
└── .gitignore               # Standard ignore file
```

---

## 🚀 How to Set Up and Run Locally

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd food-recipe-mlops
```

### 2. Install Python Dependencies

It is recommended to create a fresh environment:

```bash
conda create -n fooddetector python=3.10
conda activate fooddetector
pip install -r requirements.txt
```

**Key dependencies:**
- `fastapi`
- `uvicorn`
- `pillow`
- `ultralytics`
- `mlflow`
- `jinja2`

### 3. Start MLflow Server Locally

```bash
mlflow ui --port 5001
```

Visit the MLflow UI at: [http://127.0.0.1:5001/](http://127.0.0.1:5001/)

### 4. Train a YOLOv8 Model (If Needed)

```bash
python training/train.py
```

- ✅ Logs metrics and model automatically to MLflow.
- ✅ Saves the best model as `/runs/detect/train/weights/best.pt`.

### 5. (Optional) Log an Already Trained Model

If you already have a trained model (e.g., downloaded from Roboflow), you can manually log it:

```bash
python training/log_saved_model.py
```

- ✅ Manually logs and registers the model into the MLflow Model Registry.

### 6. Start the FastAPI Server

```bash
uvicorn app.main:app --reload
```

Visit the application at: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

- ✅ Upload a food image.
- ✅ View detected ingredients.

---

## 🧠 Key Features

- **Transfer Learning on YOLOv8**: Fast and accurate ingredient detection.
- **Ingredient Detection**: Works with real-world messy photos.
- **Simple Frontend**: Upload images easily.
- **FastAPI Backend**: Clean and scalable architecture.
- **MLflow Integration**:
    - Experiment tracking (losses, metrics).
    - Model artifact logging.
    - Model versioning.
- **Error Handling**: Prevents crashes during model logging or prediction.
- **Future Extension**: Connect to LLMs (e.g., OpenAI, Gemini) for recipe generation.

---

## 📊 Example Detected Ingredients Output

**Uploaded Image ➔**

**Detected Ingredients:**
- Tomato (98.5%)
- Bell Pepper (94.2%)
- Lettuce (89.3%)

---

## 🔥 Future Improvements

- Draw bounding boxes on the uploaded image.
- Recommend recipes based on detected ingredients.
- Add a cuisine selector (e.g., Indian, Keto, Vegan).
- Deploy to GCP with a custom domain and SSL (production-ready version).
- Add logging, monitoring, and alerts.

---

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/)
- [Roboflow Datasets](https://roboflow.com/)
