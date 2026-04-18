# MLFlow Stack — Brain MRI Segmentation

**Date:** 2026-04-18  
**Objectif:** Ajouter un stack MLFlow complet (tracking + model registry) au projet de segmentation IRM, avec infrastructure Docker locale et entraînement sur Google Colab.

---

## Contexte

Le projet entraîne deux modèles de segmentation de tumeurs cérébrales sur le dataset TCGA :
- **UNet** (via `segmentation_models_pytorch`, encoder ResNet34)
- **YOLOv8n-seg** (via Ultralytics)

L'entraînement se fait sur Google Colab (pas de GPU local). L'objectif est pédagogique : comprendre et démontrer MLFlow pour un CV MLOps.

---

## Architecture globale

```
[Local PC]                              [Google Colab]
Docker Compose:                         train_unet.py / train_yolo.py
  - MLFlow server (port 5000)   <───    mlflow.set_tracking_uri(ngrok_url)
  - PostgreSQL (metadata)               mlflow.log_params / metrics / model
  - MinIO (artifacts S3)
  - ngrok (tunnel public)
```

---

## 1. Infrastructure Docker

### Services (`docker-compose.yml`)

| Service | Image | Port | Rôle |
|---|---|---|---|
| `postgres` | `postgres:15` | 5432 | Backend metadata (runs, params, métriques) |
| `minio` | `minio/minio` | 9000/9001 | Stockage artifacts S3-compatible |
| `mlflow` | custom Dockerfile | 5000 | Serveur de tracking MLFlow |
| `ngrok` | `ngrok/ngrok` | 4040 | Tunnel HTTP public vers Colab |

### Volumes persistants
- `postgres_data` : survit aux `docker compose down`
- `minio_data` : artifacts (modèles, plots) persistants

### Fichier `.env`
Contient les credentials : `POSTGRES_PASSWORD`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, `NGROK_AUTHTOKEN`. Non commité (`.gitignore`).

---

## 2. Structure des fichiers

```
Brain-MRI-Segmentation/
├── docker-compose.yml
├── mlflow.Dockerfile
├── .env                        # credentials, non commité
├── .env.example                # template commité
├── model.py                    # (existant) architecture UNet
├── brain_tumor.yaml            # (existant) config dataset YOLO
│
├── src/
│   ├── dataset.py              # MRIDataset extrait du notebook
│   ├── train_unet.py           # Entraînement UNet + tracking MLFlow manuel
│   └── train_yolo.py           # Entraînement YOLOv8 + MLFlow autolog
│
└── notebooks/
    └── data_mapping.ipynb      # (déplacé) EDA et préparation dataset
```

---

## 3. MLFlow Tracking

### Expérience UNet : `UNet-Segmentation`

Tracking **manuel** via l'API MLFlow (démontre la maîtrise de l'API).

| Type | Paramètre / Métrique |
|---|---|
| `log_param` | `encoder`, `lr`, `batch_size`, `epochs`, `image_size`, `optimizer` |
| `log_metric` | `train_loss`, `val_loss`, `val_dice` (par époque) |
| `log_artifact` | PNG courbe loss/dice (matplotlib) |
| `log_model` | `mlflow.pytorch.log_model` → Model Registry |

### Expérience YOLOv8 : `YOLOv8-Segmentation`

Tracking via **callback MLFlow natif Ultralytics** (démontre la connaissance des intégrations tierces).

- Métriques automatiques : `mAP50`, `mAP50-95`, `precision`, `recall`, `box_loss`, `seg_loss`
- `mlflow.pytorch.log_model` sur le meilleur checkpoint en fin de run

---

## 4. Model Registry

| Modèle | Nom Registry | Métrique de sélection |
|---|---|---|
| UNet ResNet34 | `unet-brain-mri` | `val_dice` max |
| YOLOv8n-seg | `yolov8-brain-mri` | `mAP50` max |

**Workflow :**
1. Chaque run enregistre automatiquement le modèle en stage `Staging`
2. Promotion manuelle vers `Production` depuis l'UI MLFlow après comparaison des runs
3. Les deux modèles sont comparables dans l'UI (deux expériences distinctes)

---

## 5. Workflow utilisateur

```bash
# 1. Démarrer le stack local
docker compose up -d

# 2. Récupérer l'URL ngrok (depuis http://localhost:4040 ou les logs)
docker compose logs ngrok

# 3. Dans Colab : configurer l'URI de tracking
import mlflow
mlflow.set_tracking_uri("https://xxxx.ngrok-free.app")

# 4. Lancer l'entraînement
# !python src/train_unet.py
# !python src/train_yolo.py

# 5. Visualiser les résultats
# http://localhost:5000
```

---

## Hors scope

- Serving/déploiement du modèle (MLFlow Models serving)
- CI/CD automatisé
- `MLproject` file (reproductibilité MLFlow Projects)
- Authentification sécurisée du serveur MLFlow
