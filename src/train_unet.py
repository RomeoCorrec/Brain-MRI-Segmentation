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
