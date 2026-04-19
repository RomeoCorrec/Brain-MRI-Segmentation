import os
import glob
import argparse
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def mask_to_yolo_polygons(mask):
    """Return list of normalized polygon strings from a binary mask."""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2)
        coords = " ".join(f"{x/w:.6f} {y/h:.6f}" for x, y in pts)
        lines.append(f"0 {coords}")
    return lines


def prepare(data_dir, output_dir, val_ratio=0.2, seed=42):
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "*", "*_mask.tif")))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in {data_dir}")
    print(f"Found {len(mask_paths)} samples")

    train_masks, val_masks = train_test_split(mask_paths, test_size=val_ratio, random_state=seed)

    for split, masks in [("train", train_masks), ("val", val_masks)]:
        img_dir = os.path.join(output_dir, "images", split)
        lbl_dir = os.path.join(output_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for mask_path in masks:
            img_path = mask_path.replace("_mask.tif", ".tif")
            if not os.path.exists(img_path):
                continue

            stem = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path)
            if img is None:
                continue
            cv2.imwrite(os.path.join(img_dir, f"{stem}.jpg"), img)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            polygons = mask_to_yolo_polygons(mask) if mask is not None else []
            with open(os.path.join(lbl_dir, f"{stem}.txt"), "w") as f:
                f.write("\n".join(polygons))

        print(f"  {split}: {len(masks)} images written")

    yaml_path = os.path.join(output_dir, "brain_tumor.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n  0: tumor\n")
    print(f"Dataset yaml written to {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to kaggle_3m directory")
    parser.add_argument("--output-dir", default="/content/yolo_dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    prepare(args.data_dir, args.output_dir, args.val_ratio)
