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
