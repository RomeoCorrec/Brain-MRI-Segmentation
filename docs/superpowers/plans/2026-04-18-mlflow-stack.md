# MLFlow Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter un stack MLFlow complet (tracking + model registry) avec infrastructure Docker locale et scripts d'entraînement compatibles Google Colab.

**Architecture:** Un `docker-compose.yml` lance MLFlow server (PostgreSQL backend + MinIO artifacts) exposé via ngrok. Deux scripts Python (`train_unet.py`, `train_yolo.py`) se connectent au serveur distant via `--tracking-uri` et loggent params, métriques et modèles dans le registry.

**Tech Stack:** MLFlow 2.14, PostgreSQL 15, MinIO, ngrok, PyTorch, segmentation_models_pytorch, Ultralytics YOLOv8, pytest, albumentations

---

## File Map

| Fichier | Action | Responsabilité |
|---|---|---|
| `docker-compose.yml` | CREATE | Stack complet : postgres, minio, minio-init, mlflow, ngrok |
| `mlflow.Dockerfile` | CREATE | Image MLFlow avec psycopg2 + boto3 |
| `.env.example` | CREATE | Template de credentials (non-secret) |
| `.gitignore` | MODIFY | Ignorer `.env`, `mlruns/`, `best_unet.pth`, `curves.png` |
| `requirements.txt` | MODIFY | Ajouter `mlflow==2.14.3` et `boto3` |
| `src/__init__.py` | CREATE | Rendre `src/` importable |
| `src/dataset.py` | CREATE | Classe `MRIDataset` extraite du notebook |
| `src/train_unet.py` | CREATE | Entraînement UNet + MLFlow tracking manuel |
| `src/train_yolo.py` | CREATE | Entraînement YOLOv8 + MLFlow autolog Ultralytics |
| `tests/__init__.py` | CREATE | Rendre `tests/` importable |
| `tests/test_dataset.py` | CREATE | Tests unitaires `MRIDataset` et `calculate_dice` |
| `notebooks/data_mapping.ipynb` | MOVE | Depuis la racine |

---

## Task 1: Infrastructure Docker

**Files:**
- Create: `docker-compose.yml`
- Create: `mlflow.Dockerfile`
- Create: `.env.example`
- Modify: `.gitignore`

- [ ] **Step 1: Créer `mlflow.Dockerfile`**

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.14.3 psycopg2-binary boto3
```

- [ ] **Step 2: Créer `.env.example`**

```env
POSTGRES_PASSWORD=mlflow_password
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
NGROK_AUTHTOKEN=your_ngrok_token_here
```

Obtenir un token gratuit sur https://dashboard.ngrok.com/get-started/your-authtoken

- [ ] **Step 3: Créer `docker-compose.yml`**

```yaml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "mlflow"]
      interval: 5s
      retries: 5

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      retries: 5

  minio-init:
    image: minio/mc
    depends_on:
      minio:
        condition: service_healthy
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 $${MINIO_ROOT_USER} $${MINIO_ROOT_PASSWORD} &&
      mc mb --ignore-existing local/mlflow-artifacts &&
      exit 0
      "

  mlflow:
    build:
      context: .
      dockerfile: mlflow.Dockerfile
    depends_on:
      postgres:
        condition: service_healthy
      minio-init:
        condition: service_completed_successfully
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER}
      AWS_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD}
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:${POSTGRES_PASSWORD}@postgres:5432/mlflow
      --default-artifact-root s3://mlflow-artifacts/
      --host 0.0.0.0
      --port 5000

  ngrok:
    image: ngrok/ngrok:latest
    environment:
      NGROK_AUTHTOKEN: ${NGROK_AUTHTOKEN}
    command: http mlflow:5000
    ports:
      - "4040:4040"
    depends_on:
      - mlflow

volumes:
  postgres_data:
  minio_data:
```

- [ ] **Step 4: Mettre à jour `.gitignore`**

Ajouter à la fin du fichier `.gitignore` existant (ou le créer s'il n'existe pas) :

```
.env
mlruns/
best_unet.pth
curves.png
*.pt
!yolov8n-seg.pt
```

- [ ] **Step 5: Copier `.env.example` en `.env` et remplir les valeurs**

```bash
cp .env.example .env
# Éditer .env : remplacer your_ngrok_token_here par ton vrai token ngrok
```

- [ ] **Step 6: Démarrer le stack et vérifier**

```bash
docker compose up -d --build
```

Attendre ~30 secondes, puis vérifier :

```bash
docker compose ps
```

Résultat attendu : tous les services `Running` (minio-init sera `Exited (0)`, c'est normal).

```bash
docker compose logs ngrok
```

Chercher une ligne comme : `url=https://xxxx.ngrok-free.app` — c'est l'URL à utiliser dans Colab.

Ouvrir http://localhost:5000 dans le navigateur → UI MLFlow vide, OK.

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml mlflow.Dockerfile .env.example .gitignore
git commit -m "feat: add Docker infrastructure for MLFlow stack"
```

---

## Task 2: Dépendances Python

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Ajouter mlflow et boto3 à `requirements.txt`**

Ajouter ces deux lignes à la fin de `requirements.txt` :

```
mlflow==2.14.3
boto3==1.38.0
```

- [ ] **Step 2: Vérifier l'installation dans l'environnement local**

```bash
pip install mlflow==2.14.3 boto3==1.38.0
python -c "import mlflow; print(mlflow.__version__)"
```

Résultat attendu : `2.14.3`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add mlflow and boto3 dependencies"
```

---

## Task 3: Module Dataset

**Files:**
- Create: `src/__init__.py`
- Create: `src/dataset.py`
- Create: `tests/__init__.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Créer `src/__init__.py` et `tests/__init__.py`**

Les deux fichiers sont vides :

```bash
# Sur Windows PowerShell :
New-Item -ItemType File src/__init__.py
New-Item -ItemType File tests/__init__.py
```

- [ ] **Step 2: Écrire le test en premier**

Créer `tests/test_dataset.py` :

```python
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch
from src.dataset import MRIDataset, calculate_dice


def make_fake_df(tmp_path, n=4):
    """Crée de fausses images TIF et un DataFrame associé."""
    import cv2
    rows = []
    for i in range(n):
        img_path = tmp_path / f"img_{i}.tif"
        mask_path = tmp_path / f"img_{i}_mask.tif"
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        if i % 2 == 0:
            mask[20:40, 20:40] = 255
        cv2.imwrite(str(img_path), img)
        cv2.imwrite(str(mask_path), mask)
        rows.append({"image_path": str(img_path), "mask_path": str(mask_path)})
    return pd.DataFrame(rows)


def test_dataset_length(tmp_path):
    df = make_fake_df(tmp_path, n=4)
    ds = MRIDataset(df)
    assert len(ds) == 4


def test_dataset_output_shapes(tmp_path):
    df = make_fake_df(tmp_path, n=2)
    ds = MRIDataset(df)
    image, mask = ds[0]
    assert image.shape == (3, 64, 64)
    assert mask.shape == (1, 64, 64)


def test_dataset_mask_is_binary(tmp_path):
    df = make_fake_df(tmp_path, n=2)
    ds = MRIDataset(df)
    _, mask = ds[0]
    unique_vals = torch.unique(mask)
    assert all(v in [0.0, 1.0] for v in unique_vals)


def test_calculate_dice_perfect():
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    score = calculate_dice(logits, targets)
    assert abs(score - 1.0) < 1e-4


def test_calculate_dice_zero():
    logits = torch.tensor([[[[10.0, 10.0]]]])
    targets = torch.tensor([[[[0.0, 0.0]]]])
    score = calculate_dice(logits, targets)
    assert score == 0.0
```

- [ ] **Step 2: Vérifier que les tests échouent**

```bash
pytest tests/test_dataset.py -v
```

Résultat attendu : `ImportError: cannot import name 'MRIDataset' from 'src.dataset'`

- [ ] **Step 3: Créer `src/dataset.py`**

```python
import cv2
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['image_path']
        mask_path = self.df.iloc[index]['mask_path']

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        mask = mask.float()
        mask[mask > 0] = 1.0

        return image, mask


def calculate_dice(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum()
    return (2. * intersection / (preds.sum() + targets.sum() + 1e-8)).item()
```

- [ ] **Step 4: Vérifier que les tests passent**

```bash
pytest tests/test_dataset.py -v
```

Résultat attendu : `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/__init__.py src/dataset.py tests/__init__.py tests/test_dataset.py
git commit -m "feat: extract MRIDataset to src/dataset.py with tests"
```

---

## Task 4: Script d'entraînement UNet avec MLFlow

**Files:**
- Create: `src/train_unet.py`

- [ ] **Step 1: Créer `src/train_unet.py`**

```python
import os
import glob
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import mlflow
import mlflow.pytorch

from src.dataset import MRIDataset, calculate_dice


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size):
    train_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tf, val_tf


def build_dataframes(data_dir):
    mask_files = glob.glob(f'{data_dir}/*/*_mask.tif')
    data_list = [
        {'image_path': m.replace('_mask', ''), 'mask_path': m}
        for m in mask_files
    ]
    df = pd.DataFrame(data_list)
    df['patient_id'] = df['image_path'].apply(lambda x: os.path.dirname(x))
    patient_ids = df['patient_id'].unique()
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    val_ids, _ = train_test_split(temp_ids, test_size=0.5, random_state=42)
    train_df = df[df['patient_id'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def save_curves(train_losses, val_losses, val_dices, path="curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='train')
    ax1.plot(val_losses, label='val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(val_dices, label='val dice', color='green')
    ax2.set_title('Dice Score')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_tf, val_tf = get_transforms(cfg['image_size'])
    train_df, val_df = build_dataframes(cfg['data_dir'])

    train_loader = DataLoader(
        MRIDataset(train_df, train_tf),
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        MRIDataset(val_df, val_tf),
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = smp.Unet(
        encoder_name=cfg['encoder'],
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

    mlflow.set_tracking_uri(cfg['tracking_uri'])
    mlflow.set_experiment("UNet-Segmentation")

    with mlflow.start_run():
        mlflow.log_params({
            "encoder": cfg['encoder'],
            "lr": cfg['lr'],
            "batch_size": cfg['batch_size'],
            "epochs": cfg['epochs'],
            "image_size": cfg['image_size'],
            "optimizer": "adam",
        })

        train_losses, val_losses, val_dices = [], [], []
        best_val_loss = float('inf')

        for epoch in range(cfg['epochs']):
            # --- Train ---
            model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # --- Validation ---
            model.eval()
            val_loss, val_dice = 0.0, 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item()
                    val_dice += calculate_dice(outputs, masks)

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            avg_dice = val_dice / len(val_loader)

            train_losses.append(avg_train)
            val_losses.append(avg_val)
            val_dices.append(avg_dice)

            mlflow.log_metrics(
                {"train_loss": avg_train, "val_loss": avg_val, "val_dice": avg_dice},
                step=epoch,
            )
            print(
                f"Epoch {epoch+1}/{cfg['epochs']} | "
                f"train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | val_dice={avg_dice:.4f}"
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), "best_unet.pth")
                print("  -> Meilleur modèle sauvegardé")

        # --- Artifacts & Model Registry ---
        save_curves(train_losses, val_losses, val_dices, "curves.png")
        mlflow.log_artifact("curves.png")

        model.load_state_dict(torch.load("best_unet.pth", map_location=device))
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="unet-brain-mri",
        )
        print("Modèle enregistré dans le registry MLFlow sous 'unet-brain-mri'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement UNet avec MLFlow tracking")
    parser.add_argument("--tracking-uri", default="http://localhost:5000",
                        help="URI du serveur MLFlow (ex: https://xxxx.ngrok-free.app)")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    cfg = {
        "tracking_uri": args.tracking_uri,
        "encoder": args.encoder,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "data_dir": args.data_dir,
    }
    train(cfg)
```

- [ ] **Step 2: Vérifier la syntaxe du script**

```bash
python -m py_compile src/train_unet.py && echo "OK"
```

Résultat attendu : `OK`

- [ ] **Step 3: Tester le smoke test avec MLFlow local (1 époque)**

Ce test vérifie que le script tourne sans erreur avec un backend MLFlow local (aucun serveur Docker requis). Créer temporairement un fichier `tests/test_train_unet_smoke.py` :

```python
import os
import glob
import pandas as pd
import pytest
from unittest.mock import patch
import torch


def test_train_unet_smoke(tmp_path):
    """Vérifie que train() tourne 1 époque sans crash avec un faux dataset."""
    import cv2
    import numpy as np

    # Créer un faux dataset (2 patients, 2 images chacun)
    for patient in ["P001", "P002", "P003", "P004", "P005"]:
        patient_dir = tmp_path / patient
        patient_dir.mkdir()
        for i in range(2):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[10:20, 10:20] = 255
            cv2.imwrite(str(patient_dir / f"scan_{i}.tif"), img)
            cv2.imwrite(str(patient_dir / f"scan_{i}_mask.tif"), mask)

    import mlflow
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

    from src.train_unet import train
    cfg = {
        "tracking_uri": f"file://{tmp_path}/mlruns",
        "encoder": "resnet18",
        "lr": 1e-3,
        "batch_size": 2,
        "epochs": 1,
        "image_size": 64,
        "data_dir": str(tmp_path),
    }
    train(cfg)

    runs = mlflow.search_runs(experiment_names=["UNet-Segmentation"])
    assert len(runs) == 1
    assert "metrics.val_dice" in runs.columns
```

```bash
pytest tests/test_train_unet_smoke.py -v -s
```

Résultat attendu : `1 passed` (peut prendre 30-60 secondes)

- [ ] **Step 4: Supprimer le fichier de smoke test (il était temporaire)**

```bash
git rm --cached tests/test_train_unet_smoke.py 2>/dev/null; rm tests/test_train_unet_smoke.py
```

- [ ] **Step 5: Commit**

```bash
git add src/train_unet.py
git commit -m "feat: add UNet training script with MLFlow tracking"
```

---

## Task 5: Script d'entraînement YOLOv8 avec MLFlow

**Files:**
- Create: `src/train_yolo.py`

- [ ] **Step 1: Créer `src/train_yolo.py`**

```python
import os
import argparse
import mlflow
import mlflow.pytorch
from ultralytics import YOLO


def train(cfg):
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

    # Enregistrer le meilleur modèle dans le Model Registry MLFlow
    best_weights = results.save_dir / 'weights' / 'best.pt'
    best_model = YOLO(str(best_weights))

    mlflow.set_tracking_uri(cfg['tracking_uri'])
    mlflow.set_experiment("YOLOv8-Segmentation")

    with mlflow.start_run(run_name="model-registration"):
        mlflow.log_param("source_weights", str(best_weights))
        mlflow.pytorch.log_model(
            best_model.model,
            artifact_path="model",
            registered_model_name="yolov8-brain-mri",
        )
    print("Modèle enregistré dans le registry MLFlow sous 'yolov8-brain-mri'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement YOLOv8 avec MLFlow tracking")
    parser.add_argument("--tracking-uri", default="http://localhost:5000",
                        help="URI du serveur MLFlow (ex: https://xxxx.ngrok-free.app)")
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--data", default="brain_tumor.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    cfg = {
        "tracking_uri": args.tracking_uri,
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
    }
    train(cfg)
```

- [ ] **Step 2: Vérifier la syntaxe**

```bash
python -m py_compile src/train_yolo.py && echo "OK"
```

Résultat attendu : `OK`

- [ ] **Step 3: Commit**

```bash
git add src/train_yolo.py
git commit -m "feat: add YOLOv8 training script with MLFlow autolog"
```

---

## Task 6: Réorganisation du notebook

**Files:**
- Move: `data_mapping.ipynb` → `notebooks/data_mapping.ipynb`

- [ ] **Step 1: Créer le dossier `notebooks/` et déplacer le notebook**

```bash
mkdir notebooks
git mv data_mapping.ipynb notebooks/data_mapping.ipynb
```

- [ ] **Step 2: Vérifier que le fichier est bien déplacé**

```bash
git status
```

Résultat attendu : `renamed: data_mapping.ipynb -> notebooks/data_mapping.ipynb`

- [ ] **Step 3: Commit**

```bash
git add notebooks/
git commit -m "refactor: move notebook to notebooks/ directory"
```

---

## Vérification finale : workflow complet

- [ ] **Step 1: Démarrer le stack Docker (si pas déjà démarré)**

```bash
docker compose up -d
```

- [ ] **Step 2: Récupérer l'URL ngrok**

```bash
docker compose logs ngrok | grep "url="
```

Ou ouvrir http://localhost:4040 dans le navigateur.

- [ ] **Step 3: Dans Google Colab — lancer l'entraînement UNet**

```python
# Cellule 1 : Cloner le repo et installer les dépendances
!git clone https://github.com/TON_REPO/Brain-MRI-Segmentation.git
%cd Brain-MRI-Segmentation
!pip install -r requirements.txt

# Cellule 2 : Lancer le training UNet
!python src/train_unet.py \
  --tracking-uri https://xxxx.ngrok-free.app \
  --epochs 30 \
  --encoder resnet34 \
  --batch-size 16
```

- [ ] **Step 4: Dans Google Colab — lancer l'entraînement YOLOv8**

```python
!python src/train_yolo.py \
  --tracking-uri https://xxxx.ngrok-free.app \
  --epochs 10 \
  --batch 16
```

- [ ] **Step 5: Vérifier dans l'UI MLFlow**

Ouvrir http://localhost:5000 :
- Deux expériences visibles : `UNet-Segmentation` et `YOLOv8-Segmentation`
- Chaque run contient ses params, ses métriques (graphes par époque), et ses artifacts
- Aller dans **Models** → deux modèles : `unet-brain-mri` et `yolov8-brain-mri` en version `1` / `Staging`
- Promouvoir manuellement le meilleur run en `Production` via l'UI

---

## Résumé des commandes Colab utiles

```python
# Vérifier la connexion au serveur MLFlow depuis Colab
import mlflow
mlflow.set_tracking_uri("https://xxxx.ngrok-free.app")
client = mlflow.tracking.MlflowClient()
print([e.name for e in client.search_experiments()])
# Attendu : ['UNet-Segmentation', 'YOLOv8-Segmentation', 'Default']
```
