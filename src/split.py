"""Single source of truth for the train/validation split.

Every pipeline (UNet training, YOLO dataset preparation, common evaluation) must
build its split here so the three agree on which patients are held out. The split
is done at the *patient* level: slices from one patient are highly correlated, so
splitting by slice leaks near-duplicates into validation and inflates the scores.
"""
import os
import glob

import pandas as pd
from sklearn.model_selection import train_test_split


def build_dataframes(data_dir, test_size=0.2, random_state=42):
    """Return (train_df, val_df) split by patient.

    Each row has 'image_path', 'mask_path', 'patient_id'. The glob order is left
    untouched on purpose: changing it would move patients between train and val
    and silently invalidate any model already trained against the old split.
    """
    mask_files = glob.glob(f'{data_dir}/*/*_mask.tif')
    data_list = [
        {'image_path': m.replace('_mask', ''), 'mask_path': m}
        for m in mask_files
    ]
    df = pd.DataFrame(data_list)
    if df.empty:
        raise ValueError(f"No mask files found under '{data_dir}/*/*_mask.tif'. Check --data-dir.")
    df['patient_id'] = df['image_path'].apply(lambda x: os.path.dirname(x))
    patient_ids = df['patient_id'].unique()
    train_ids, val_ids = train_test_split(patient_ids, test_size=test_size, random_state=random_state)
    train_df = df[df['patient_id'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df
