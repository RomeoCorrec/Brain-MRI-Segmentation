# Brain MRI Segmentation

Deep learning project for brain tumor segmentation on the [TCGA dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation), with full MLFlow tracking and model registry.

Two models are trained and compared:
- **UNet** (ResNet34 encoder via `segmentation_models_pytorch`) — manual MLFlow tracking API
- **YOLOv8n-seg** (Ultralytics) — MLFlow autolog via native Ultralytics callback

Training runs on **Google Colab** (GPU). Experiment tracking is served locally via a **Docker Compose** stack exposed to Colab through an **ngrok** tunnel.

---

## Architecture

```
[Local PC]                              [Google Colab]
Docker Compose:                         src/train_unet.py
  - MLFlow server (port 5000)  <─────   src/train_yolo.py
  - PostgreSQL (metadata)
  - MinIO (artifacts, S3-compatible)      mlflow.set_tracking_uri(ngrok_url)
  - ngrok (public tunnel)
```

---

## Project Structure

```
Brain-MRI-Segmentation/
├── docker-compose.yml          # Full stack: postgres, minio, mlflow, ngrok
├── mlflow.Dockerfile           # MLFlow server image
├── .env.example                # Credentials template (copy to .env)
├── requirements.txt            # Direct Python dependencies
│
├── src/
│   ├── dataset.py              # MRIDataset + calculate_dice
│   ├── train_unet.py           # UNet training + manual MLFlow tracking
│   └── train_yolo.py           # YOLOv8 training + MLFlow autolog
│
├── tests/
│   └── test_dataset.py         # Unit tests for dataset and dice metric
│
├── notebooks/
│   └── data_mapping.ipynb      # EDA and dataset preparation
│
├── model.py                    # UNet architecture
└── brain_tumor.yaml            # YOLO dataset config
```

---

## MLFlow Tracking

| Experiment | Model | Tracking method |
|---|---|---|
| `UNet-Segmentation` | UNet ResNet34 | Manual API (`log_params`, `log_metrics`, `log_artifact`) |
| `YOLOv8-Segmentation` | YOLOv8n-seg | Ultralytics native callback |

Both models are registered in the **MLFlow Model Registry**:
- `unet-brain-mri` — promoted by best `val_dice`
- `yolov8-brain-mri` — promoted by best `mAP50`

---

## Local Setup

### Prerequisites
- Docker Desktop
- A free [ngrok account](https://dashboard.ngrok.com/get-started/your-authtoken) (for the authtoken)

### 1. Configure credentials

```bash
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD, MINIO_ROOT_PASSWORD, NGROK_AUTHTOKEN
```

### 2. Start the stack

```bash
docker compose up -d --build
docker compose ps
# All services Running (minio-init exits 0 — normal)
```

### 3. Get the ngrok URL

```bash
docker compose logs ngrok | grep "url="
# → url=https://xxxx.ngrok-free.app
```

Open **http://localhost:5000** to access the MLFlow UI.

---

## Training on Google Colab

### Setup

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/RomeoCorrec/Brain-MRI-Segmentation.git
%cd Brain-MRI-Segmentation

!pip install -q segmentation-models-pytorch==0.5.0 ultralytics==8.3.239 \
    albumentations==2.0.8 mlflow==2.14.3 boto3==1.38.0
```

### Verify connection

```python
import mlflow
mlflow.set_tracking_uri("https://xxxx.ngrok-free.app")
client = mlflow.tracking.MlflowClient()
print([e.name for e in client.search_experiments()])
```

### Train UNet

```bash
!python src/train_unet.py \
  --tracking-uri https://xxxx.ngrok-free.app \
  --data-dir /content/drive/MyDrive/TCGA_dataset \
  --encoder resnet34 \
  --epochs 30 \
  --batch-size 16 \
  --output-dir /content/outputs
```

### Train YOLOv8

```bash
!python src/train_yolo.py \
  --tracking-uri https://xxxx.ngrok-free.app \
  --epochs 10 \
  --batch 16
```

---

## Model Registry workflow

1. Each run automatically registers the model in **Staging**
2. Compare runs in the MLFlow UI (Experiments → select runs → Compare)
3. Manually promote the best run to **Production** via the UI (Models → version → Stage → Production)

---

## Tests

```bash
pytest tests/ -v
```
