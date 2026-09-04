import cv2
import numpy as np

from src.split import build_dataframes


def _make_dataset(root, n_patients=10, slices_per_patient=8):
    for p in range(n_patients):
        pdir = root / f"TCGA_XX_{p:02d}"
        pdir.mkdir()
        for s in range(slices_per_patient):
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            mask = np.zeros((32, 32), dtype=np.uint8)
            cv2.imwrite(str(pdir / f"slice_{s}.tif"), img)
            cv2.imwrite(str(pdir / f"slice_{s}_mask.tif"), mask)


def test_split_is_by_patient_and_disjoint(tmp_path):
    _make_dataset(tmp_path)
    train_df, val_df = build_dataframes(str(tmp_path))

    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])

    # no patient appears in both splits -> no slice-level leakage
    assert train_patients.isdisjoint(val_patients)
    # every slice of a patient lands in exactly one split
    assert len(train_df) + len(val_df) == 10 * 8
    assert len(val_patients) == 2  # 20% of 10 patients


def test_split_is_deterministic(tmp_path):
    _make_dataset(tmp_path)
    a_train, a_val = build_dataframes(str(tmp_path))
    b_train, b_val = build_dataframes(str(tmp_path))
    assert set(a_val["patient_id"]) == set(b_val["patient_id"])
