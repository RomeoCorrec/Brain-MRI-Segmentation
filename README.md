# Brain MRI Segmentation

Deep learning project for brain tumor segmentation on the [TCGA LGG dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation), with full MLFlow tracking and model registry.

Two models are trained and compared:
- **UNet** (ResNet34 encoder via `segmentation_models_pytorch`) — manual MLFlow tracking API
- **YOLOv8n-seg** (Ultralytics) — MLFlow autolog via native Ultralytics callback

Training runs on **Google Colab** (GPU). Experiment tracking is served locally via a **Docker Compose** stack (MLFlow + PostgreSQL) exposed to Colab through an **ngrok** tunnel.

---

## Architecture

```
[Local PC]                              [Google Colab]
Docker Compose:                         notebooks/colab_training.ipynb
  - MLFlow server (port 5000)  <─────     src/train_unet.py
  - PostgreSQL (metadata)                 src/train_yolo.py
  - ngrok (public tunnel)
                                          mlflow.set_tracking_uri(ngrok_url)
```

Artifacts (model weights, curves) are stored on the MLFlow server's local filesystem and served via `--serve-artifacts`, so Colab only needs the ngrok URL — no S3/MinIO credentials required.

---

## Project Structure

```
Brain-MRI-Segmentation/
├── docker-compose.yml          # MLFlow stack: postgres, mlflow, ngrok
├── mlflow.Dockerfile           # MLFlow server image
├── requirements.txt            # Python dependencies
│
├── src/
│   ├── dataset.py              # MRIDataset + calculate_dice
│   ├── prepare_yolo_dataset.py # Convert TIFF masks → YOLO polygon format
│   ├── train_unet.py           # UNet training + manual MLFlow tracking
│   └── train_yolo.py           # YOLOv8 training + MLFlow autolog
│
├── tests/
│   └── test_dataset.py         # Unit tests for dataset and dice metric
│
└── notebooks/
    └── colab_training.ipynb    # Full Colab pipeline (sections 1–7)
```

> `brain_tumor.yaml` is generated at runtime by `prepare_yolo_dataset.py` and is not tracked in git.

---

## MLFlow Tracking

| Experiment | Model | Tracking method |
|---|---|---|
| `UNet-Segmentation` | UNet ResNet34 | Manual API (`log_params`, `log_metrics`, `log_artifact`) |
| `YOLOv8-Segmentation` | YOLOv8n-seg | Ultralytics native callback |

Logged metrics:
- **UNet**: `train_loss`, `val_loss`, `val_dice` (per epoch)
- **YOLOv8**: `metrics/mAP50M`, `metrics/mAP50-95M`, `metrics/precisionM`, `metrics/recallM`, seg losses (per epoch)

Both models are registered in the **MLFlow Model Registry**:
- `unet-brain-mri` — best checkpoint by `val_loss`
- `yolov8-brain-mri` — best checkpoint by `mAP50`

---

## Local Setup

### Prerequisites
- Docker Desktop
- A free [ngrok account](https://dashboard.ngrok.com/get-started/your-authtoken)

### 1. Configure credentials

```bash
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD and NGROK_AUTHTOKEN
```

### 2. Start the stack

```bash
docker compose up -d --build
docker compose ps
# All services should be Running
```

### 3. Get the ngrok URL

```bash
docker compose logs ngrok | grep "url="
# → url=https://xxxx.ngrok-free.app
```

Open **http://localhost:5000** to access the MLFlow UI locally.

---

## Training on Google Colab

The full pipeline is in `notebooks/colab_training.ipynb`. Run cells section by section:

| Section | Content |
|---|---|
| 1 | Install dependencies + numpy fix + runtime restart |
| 2 | Clone repo, set config (ngrok URL, Kaggle credentials) |
| 3 | Download and extract the TCGA dataset from Kaggle |
| 4 | Train UNet (30 epochs) |
| 5 | Prepare YOLO dataset (TIFF masks → polygon labels) |
| 6 | Train YOLOv8n-seg (10 epochs) |
| 7 | Compare models: training curves, metrics table, visual predictions |

> **Important**: After section 1 (pip installs), you must restart the Colab runtime (`Runtime > Restart session`) before continuing. This is required to avoid a numpy binary incompatibility between ultralytics and pandas.

### Quick start

```python
# Section 2 — config cell
NGROK_URL    = "https://xxxx.ngrok-free.app"  # from docker compose logs ngrok
KAGGLE_USER  = "your_kaggle_username"
KAGGLE_KEY   = "your_kaggle_api_key"
```

---

## Model Comparison (Section 7)

After both trainings complete, section 7 generates:

1. **Training curves** — loss and metrics per epoch for both models side by side
2. **Summary table** — best val_dice (UNet) vs best mAP50 (YOLOv8) and other key metrics
3. **Visual predictions** — UNet mask overlay vs YOLO polygon overlay on validation images with confirmed tumors

---

## Model Registry Workflow

1. Each training run registers the best checkpoint in the MLFlow registry under **Staging**
2. Compare runs in the MLFlow UI (`Experiments → select runs → Compare`)
3. Promote the best run to **Production** via `Models → version → Stage → Production`

---

## Tests

```bash
pytest tests/ -v
```
