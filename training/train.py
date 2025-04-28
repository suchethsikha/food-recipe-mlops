from ultralytics import YOLO
import mlflow
import os
from mlflow import MlflowClient
from ultralytics.utils import SETTINGS 

SETTINGS['mlflow'] = False


def train_model(data_yaml_path, model_type="yolov8n.pt", epochs=50, imgsz=640):
    print("Setting MLflow Tracking URI...")
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("ingredient-detection")

    with mlflow.start_run() as run:
        model = YOLO(model_type)
        print(f"Loaded model: {model_type}")
        print("Training on data from", data_yaml_path)

        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            pretrained=True,
            verbose=True
        )

        # ---------------------------
        # ✅ Correct way to get metrics
        # ---------------------------
        try:
            box_metrics = results.box
            metrics_to_log = {
                "precision": box_metrics.mp,      # mean precision
                "recall": box_metrics.mr,          # mean recall
                "map50": box_metrics.map50,        # mean AP @ IoU=0.5
                "map50-95": box_metrics.map,       # mean AP @ IoU=0.5:0.95
            }
            mlflow.log_metrics(metrics_to_log)
            print(f"✅ Logged metrics: {metrics_to_log}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to log metrics to MLflow: {e}")

        # ---------------------------
        # ✅ Correct way to save model
        # ---------------------------
        try:
            # Find saved best model manually
            # Ultralytics automatically saves best.pt at:
            # 'runs/detect/train7/weights/best.pt'
            # We find this path programmatically:
            best_model_path = os.path.join(results.save_dir, "weights", "best.pt")

            if os.path.exists(best_model_path):
                artifact_dir = "best_model"
                os.makedirs(artifact_dir, exist_ok=True)
                new_path = os.path.join(artifact_dir, "best.pt")
                os.system(f"cp '{best_model_path}' '{new_path}'")
                mlflow.log_artifacts(artifact_dir)

                # Register Model
                client = MlflowClient()
                run_id = run.info.run_id
                model_uri = f"runs:/{run_id}/{artifact_dir}"

                registered_model_name = "food-ingredient-detector"

                model_version = client.create_model_version(
                    name=registered_model_name,
                    source=model_uri,
                    run_id=run_id
                )

                print(f"✅ Model registered successfully: {registered_model_name} v{model_version.version}")

        except Exception as e:
            print(f"⚠️ Warning: Failed to register model to MLflow: {e}")

        print("✅ Training loop finished safely!")

if __name__ == "__main__":
    train_model(
        data_yaml_path="data/FOOD-INGREDIENTS detection.v1i.yolov12/data.yaml",
        model_type="yolov8n.pt",
        epochs=2,
        imgsz=640
    )
