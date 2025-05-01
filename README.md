# 🥘 Food Ingredient Detector and Recipe Recommender
This project uses a **YOLOv8 object detection model** trained on food/ingredient images to:

- Detect ingredients from a **photo** uploaded by the user.
- Recommend recipes based on detected ingredients using an LLM.

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
│   ├── llm_recommender_vertex.py
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
conda create -n fooddetector python=3.12.10
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


---

## Gemini Recipe Generator with Vertex AI

This project uses Google Cloud's Vertex AI Gemini API to generate Michelin-level recipes based on a list of detected ingredients.

---

### Prerequisites

- A Google Cloud project with **Vertex AI API enabled**
- Python 3.8+
- `pip` package manager
- Access to create and download a **Service Account JSON key**


---

### Set Up Google Cloud Authentication

1. **Create a service account and download its key:**
   - Go to [Google Cloud Console – IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
   - Create or select a service account.
   - Grant it the **Vertex AI User** role.
   - Click three dots next to the account -> manage keys -> Add Key -> **“Create New Key”** → Select **JSON** → Download the file.
   - Save it somewhere safe on your local machine.

---

2. **Set the required environment variables in your terminal** before running the script:

   ```bash
   export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   export GOOGLE_CLOUD_LOCATION=your-region  # e.g. us-central1
   export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/your/service-account-key.json
   ```

   Example:
   ```bash
   export GOOGLE_CLOUD_PROJECT=my-genai-project
   export GOOGLE_CLOUD_LOCATION=us-west2
   export GOOGLE_APPLICATION_CREDENTIALS=/Users/you/keys/genai-service-account.json
   ```

   > 💡 You must run these `export` commands **in the same terminal session** where you'll execute the Python script, or add them to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`) to make them permanent.

---



### 🛠 Troubleshooting

- If you get an error about authentication or credentials:
  - Double check the path to your `.json` key file
  - Make sure you ran `load_dotenv()` in your script
  - Confirm the `GOOGLE_APPLICATION_CREDENTIALS` path is absolute, not relative
- Ensure the Gemini model you’re calling (`gemini-2.0-flash-001`) is available in your selected region

---


### 6. Start the FastAPI Server

```bash
uvicorn app.main:app --reload
```

Visit the application at: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

- ✅ Upload a food image.
- ✅ View detected ingredients.


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

## 🔥 Stuff to do / Task list

- Add a LLM either just to generate recipies/ to act as a chatbot.
- Check and update model to perfect inferences -- local 
- Run with ML flow running in GCP to track model training. Already have a sample ML Flow server running in GCP, will need to change URI. Quick win.
- Check how to host the entire infrastructure since it's all local, do we move it to GCP, metaflow etc.
- Currently the dataset is from RoboFlow.
- Training.py is updated but not logged properly in ML . 2 epochs take around 8 hours to run, will have to rerun sometime overnight.

---

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/)
- [Roboflow Datasets](https://roboflow.com/)
