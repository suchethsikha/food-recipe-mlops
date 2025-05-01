import os, json, logging
import mlflow
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions
import re

load_dotenv()

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

_client = genai.Client(
    vertexai=True,
    project=PROJECT,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1")
)

def generate_recipes(detected: list[str]) -> dict:
    prompt = (
        "You are a Michelin-level chef.\n"
        f"Ingredients: {', '.join(detected)}\n"
        "Suggest 3 recipes as JSON: "
        "{'recipes':[{'name':..., 'ingredients':[...], 'steps':[...], 'time_minutes':int}]}"
    )

    logging.info("LLM prompt: %s", prompt)
    # Remove try/except so you see raw errors:
    response = _client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )
    logging.info("Raw response object: %s", response)
    text = response.text.strip()
    #logging.info("LLM returned text: %s", text)
    #print("⭑ RAW LLM OUTPUT ⭑", repr(text))
    # Remove markdown fences ```json … ```
    # 1) Strip leading ```json or ``` 
    text = re.sub(r"^```(?:json)?\s*", "", text)
    # 2) Strip trailing ```
    text = re.sub(r"\s*```$", "", text)
    logging.info("Sanitized LLM output: %s", text)


    return json.loads(text)



