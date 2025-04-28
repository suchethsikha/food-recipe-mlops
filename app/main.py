# app/main.py

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.predictor import predict_image

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    predictions = predict_image(contents)
    
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "predictions": predictions
    })
