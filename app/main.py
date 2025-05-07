from typing import List
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi import HTTPException


from app.predictor import predict_image
from app.llm_recommender_vertex import generate_recipes

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

class RecommendRequest(BaseModel):
    ingredients: List[str]
    cuisine: str
    spiciness: int

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/predict")
async def predict(request: Request, files: List[UploadFile] = File(...), ajax: bool = Form(False), 
    ):
    top_predictions = []

    for file in files:
        contents = await file.read()
        predictions = predict_image(contents)

        if predictions:
            # Select the one with the highest confidence
            name, conf = max(predictions, key=lambda x: x[1])
            top_predictions.append(name)
    if ajax:
        # return just the ingredient names as JSON
        return JSONResponse({"ingredients": top_predictions})

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "predictions": top_predictions
    })

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    """
    Now FastAPI will parse:
      { "ingredients": [...], "cuisine": "...", "spiciness": 5 }
    into `req` automatically.
    """
    try:
        recipes = generate_recipes(
            detected=req.ingredients,
            cuisine=req.cuisine,
            spiciness=req.spiciness
        )
    except Exception as e:
        # optional: wrap LLM or JSON errors
        raise HTTPException(500, str(e))
    return JSONResponse(content=recipes)
