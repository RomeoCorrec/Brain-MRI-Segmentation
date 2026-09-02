import os
import shutil
import argparse
import mlflow
import mlflow.pytorch
from ultralytics import YOLO


def train(cfg):
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # Ultralytics détecte automatiquement MLFLOW_TRACKING_URI et MLFLOW_EXPERIMENT_NAME
    os.environ["MLFLOW_TRACKING_URI"] = cfg['tracking_uri']
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "YOLOv8-Segmentation"

    model = YOLO(cfg['model'])

    results = model.train(
        data=cfg['data'],
        epochs=cfg['epochs'],
        imgsz=cfg['imgsz'],
        batch=cfg['batch'],
        name='mri_yolo_experiment',
        project='runs',
    )

    # Copie déterministe des poids (le layout de save_dir varie selon la version d'ultralytics)
    best_weights = results.save_dir / 'weights' / 'best.pt'
    final_path = os.path.join(cfg['output_dir'], "best_yolo.pt")
    shutil.copy(str(best_weights), final_path)
    print(f"Poids YOLO copiés vers {final_path}")

    # Enregistrer le meilleur modèle dans le Model Registry MLFlow (non bloquant)
    best_model = YOLO(final_path)
    mlflow.set_tracking_uri(cfg['tracking_uri'])
    mlflow.set_experiment("YOLOv8-Segmentation")
    autolog_run = mlflow.last_active_run()
    try:
        with mlflow.start_run(run_name="model-registration"):
            if autolog_run:
                mlflow.set_tag("source_run_id", autolog_run.info.run_id)
            mlflow.log_param("source_weights", str(best_weights))
            mlflow.pytorch.log_model(
                best_model.model,
                artifact_path="model",
                registered_model_name="yolov8-brain-mri",
            )
        print("Modèle enregistré dans le registry MLFlow sous 'yolov8-brain-mri'")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] enregistrement registry ignoré (client/serveur MLflow ?) : {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement YOLOv8 avec MLFlow tracking")
    parser.add_argument("--tracking-uri", default="http://localhost:5000",
                        help="URI du serveur MLFlow (ex: https://xxxx.ngrok-free.app)")
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--data", default="brain_tumor.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output-dir", default=".", help="Directory for saving best_yolo.pt")
    args = parser.parse_args()

    cfg = {
        "tracking_uri": args.tracking_uri,
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "output_dir": args.output_dir,
    }
    train(cfg)
