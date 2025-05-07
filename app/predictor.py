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
