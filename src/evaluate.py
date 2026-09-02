"""Common-metric evaluation and edge-deployment profiling for UNet vs YOLOv8-seg.

Both models are scored on the SAME patient-level validation split with the SAME
pixel-level metrics (Dice, IoU), then exported to ONNX and profiled for latency /
memory so the comparison reflects both accuracy and deployment cost.

Usage (Colab, after both trainings):
    python src/evaluate.py \
        --data-dir /content/data/kaggle_3m \
        --unet-weights /content/outputs/best_unet.pth \
        --yolo-weights /content/Brain-MRI-Segmentation/runs/mri_yolo_experiment/weights/best.pt \
        --tracking-uri $TRACKING_URI \
        --output-dir /content/eval
"""
import os
import sys
import json
import time
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.train_unet import build_dataframes, IMAGENET_MEAN, IMAGENET_STD

IMG_SIZE = 256
EPS = 1e-7


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def dice_iou(pred: np.ndarray, gt: np.ndarray):
    """pred, gt: binary uint8/bool arrays of identical shape."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    dice = (2 * inter + EPS) / (p.sum() + g.sum() + EPS)
    iou = (inter + EPS) / (union + EPS)
    return float(dice), float(iou)


# --------------------------------------------------------------------------- #
# Pre / post processing
# --------------------------------------------------------------------------- #
def load_image(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def preprocess_unet(rgb):
    img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)


def load_gt_mask(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Latency profiling
# --------------------------------------------------------------------------- #
def profile_torch(model, sample, device, n_warmup=5, n_iter=50):
    model.eval()
    x = sample.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_iter
    return dt * 1000.0  # ms


def try_export_onnx(model, sample, path, device):
    try:
        torch.onnx.export(
            model.to(device).eval(), sample.to(device), path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        size_mb = os.path.getsize(path) / 1e6
        return {"onnx_path": path, "onnx_size_mb": round(size_mb, 2)}
    except Exception as e:  # noqa: BLE001
        return {"onnx_error": str(e)[:300]}


def profile_onnx(path, sample_np, n_warmup=5, n_iter=50):
    try:
        import onnxruntime as ort
    except ImportError:
        return {"onnxruntime": "not installed"}
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for _ in range(n_warmup):
        sess.run(None, {name: sample_np})
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = sess.run(None, {name: sample_np})
    dt = (time.perf_counter() - t0) / n_iter
    return {"onnx_cpu_latency_ms": round(dt * 1000.0, 2), "onnx_out_shape": list(np.shape(out[0]))}


# --------------------------------------------------------------------------- #
# Model runners
# --------------------------------------------------------------------------- #
def build_unet(encoder, weights, device):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=encoder, encoder_weights=None,
                     in_channels=3, classes=1, activation=None)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


def unet_predict(model, rgb, device):
    x = preprocess_unet(rgb).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def yolo_predict(model, img_path):
    r = model.predict(img_path, imgsz=IMG_SIZE, conf=0.25, verbose=False)[0]
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    if r.masks is not None:
        for md in r.masks.data:
            m = cv2.resize(md.cpu().numpy(), (IMG_SIZE, IMG_SIZE))
            mask = np.maximum(mask, (m > 0.5).astype(np.uint8))
    return mask


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def evaluate(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print(f"Device: {device}")

    _, val_df = build_dataframes(cfg["data_dir"])
    print(f"Patient-level validation set: {len(val_df)} slices")

    from ultralytics import YOLO
    unet = build_unet(cfg["encoder"], cfg["unet_weights"], device)
    yolo = YOLO(cfg["yolo_weights"])

    per_image = []
    qualitative = []  # (rgb, gt, unet, yolo) for a few tumor slices
    for _, row in val_df.iterrows():
        rgb = load_image(row["image_path"])
        gt = load_gt_mask(row["mask_path"])
        u = unet_predict(unet, rgb, device)
        y = yolo_predict(yolo, row["image_path"])
        du, iu = dice_iou(u, gt)
        dy, iy = dice_iou(y, gt)
        rec = {"image": os.path.basename(row["image_path"]),
               "has_tumor": bool(gt.sum() > 0),
               "unet_dice": du, "unet_iou": iu,
               "yolo_dice": dy, "yolo_iou": iy}
        per_image.append(rec)
        if gt.sum() > 0 and len(qualitative) < cfg["n_qualitative"]:
            qualitative.append((cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)), gt * 255, u * 255, y * 255))

    arr = {k: np.array([r[k] for r in per_image]) for k in
           ["unet_dice", "unet_iou", "yolo_dice", "yolo_iou"]}
    tumor = np.array([r["has_tumor"] for r in per_image])

    def agg(key, mask=None):
        v = arr[key] if mask is None else arr[key][mask]
        return {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4)}

    # Latency / size
    sample = preprocess_unet(np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8))
    sample_np = sample.numpy()
    unet_params = sum(p.numel() for p in unet.parameters())
    yolo_params = sum(p.numel() for p in yolo.model.parameters())

    unet_lat = profile_torch(unet, sample, device)
    yolo_lat = profile_torch(yolo.model, sample, device)

    onnx_unet = os.path.join(cfg["output_dir"], "unet.onnx")
    onnx_info = try_export_onnx(unet, sample, onnx_unet, device)
    if "onnx_path" in onnx_info:
        onnx_info.update(profile_onnx(onnx_unet, sample_np))

    yolo_onnx_info = {}
    try:
        p = yolo.export(format="onnx", imgsz=IMG_SIZE, opset=17)
        yolo_onnx_info = {"onnx_path": str(p),
                          "onnx_size_mb": round(os.path.getsize(p) / 1e6, 2)}
        yolo_onnx_info.update(profile_onnx(str(p), sample_np))
    except Exception as e:  # noqa: BLE001
        yolo_onnx_info = {"onnx_error": str(e)[:300]}

    summary = {
        "device": device,
        "n_val_slices": len(per_image),
        "n_val_tumor_slices": int(tumor.sum()),
        "note": "YOLO trained on an image-level split (prepare_yolo_dataset.py); "
                "some of these patient-level val slices may have been in YOLO's train set. "
                "Fix: make prepare_yolo_dataset reuse the patient-level split.",
        "unet": {
            "params_M": round(unet_params / 1e6, 2),
            "dice_all": agg("unet_dice"), "iou_all": agg("unet_iou"),
            "dice_tumor_only": agg("unet_dice", tumor), "iou_tumor_only": agg("unet_iou", tumor),
            "torch_latency_ms": round(unet_lat, 2),
            **onnx_info,
        },
        "yolo": {
            "params_M": round(yolo_params / 1e6, 2),
            "dice_all": agg("yolo_dice"), "iou_all": agg("yolo_iou"),
            "dice_tumor_only": agg("yolo_dice", tumor), "iou_tumor_only": agg("yolo_iou", tumor),
            "torch_latency_ms": round(yolo_lat, 2),
            **yolo_onnx_info,
        },
    }

    with open(os.path.join(cfg["output_dir"], "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Markdown table for slides
    md = _markdown_table(summary)
    with open(os.path.join(cfg["output_dir"], "eval_table.md"), "w") as f:
        f.write(md)
    print("\n" + md)

    _plot_qualitative(qualitative, os.path.join(cfg["output_dir"], "qualitative.png"))
    _plot_dice_hist(arr, tumor, os.path.join(cfg["output_dir"], "dice_distribution.png"))

    if cfg["tracking_uri"]:
        _log_mlflow(cfg["tracking_uri"], summary, cfg["output_dir"])

    return summary


def _markdown_table(s):
    u, y = s["unet"], s["yolo"]
    lines = [
        "| Métrique (val commune) | UNet-ResNet34 | YOLOv8n-seg |",
        "|---|---|---|",
        f"| Paramètres | {u['params_M']} M | {y['params_M']} M |",
        f"| Dice (toutes coupes) | {u['dice_all']['mean']} | {y['dice_all']['mean']} |",
        f"| Dice (coupes avec tumeur) | {u['dice_tumor_only']['mean']} | {y['dice_tumor_only']['mean']} |",
        f"| IoU (coupes avec tumeur) | {u['iou_tumor_only']['mean']} | {y['iou_tumor_only']['mean']} |",
        f"| Latence PyTorch ({s['device']}) | {u['torch_latency_ms']} ms | {y['torch_latency_ms']} ms |",
        f"| Latence ONNX (CPU) | {u.get('onnx_cpu_latency_ms', 'n/a')} ms | {y.get('onnx_cpu_latency_ms', 'n/a')} ms |",
        f"| Taille ONNX | {u.get('onnx_size_mb', 'n/a')} MB | {y.get('onnx_size_mb', 'n/a')} MB |",
    ]
    return "\n".join(lines) + "\n"


def _plot_qualitative(items, path):
    if not items:
        return
    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    for ax, t in zip(axes[0], ["Image", "Vérité terrain", "UNet", "YOLOv8"]):
        ax.set_title(t)
    for r, (rgb, gt, u, y) in enumerate(items):
        for c, im in enumerate([rgb, gt, u, y]):
            axes[r, c].imshow(im, cmap=None if c == 0 else "gray")
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def _plot_dice_hist(arr, tumor, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(arr["unet_dice"][tumor], bins=20, alpha=0.6, label="UNet", color="steelblue")
    ax.hist(arr["yolo_dice"][tumor], bins=20, alpha=0.6, label="YOLOv8", color="salmon")
    ax.set_xlabel("Dice (coupes avec tumeur)")
    ax.set_ylabel("Nombre de coupes")
    ax.set_title("Distribution du Dice par coupe")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def _log_mlflow(uri, summary, output_dir):
    import mlflow
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("Model-Comparison")
    with mlflow.start_run(run_name="common-metric-evaluation"):
        for model_key in ("unet", "yolo"):
            m = summary[model_key]
            mlflow.log_metric(f"{model_key}_dice_tumor", m["dice_tumor_only"]["mean"])
            mlflow.log_metric(f"{model_key}_iou_tumor", m["iou_tumor_only"]["mean"])
            mlflow.log_metric(f"{model_key}_latency_ms", m["torch_latency_ms"])
            mlflow.log_metric(f"{model_key}_params_M", m["params_M"])
        for fn in ("eval_summary.json", "eval_table.md", "qualitative.png", "dice_distribution.png"):
            p = os.path.join(output_dir, fn)
            if os.path.exists(p):
                mlflow.log_artifact(p)
    print("Logged to MLflow experiment 'Model-Comparison'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Common-metric evaluation + ONNX profiling")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--unet-weights", required=True)
    ap.add_argument("--yolo-weights", required=True)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--tracking-uri", default="")
    ap.add_argument("--output-dir", default="eval")
    ap.add_argument("--n-qualitative", type=int, default=5)
    args = ap.parse_args()
    evaluate(vars(args))
