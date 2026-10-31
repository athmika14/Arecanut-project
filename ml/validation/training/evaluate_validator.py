# -*- coding: utf-8 -*-
"""
evaluate_validator.py — ArecaNut Validator: Test Set Evaluation
================================================================
Loads the best trained validator model and evaluates it on the held-out
test partition.

Reports:
  - Accuracy, precision, recall, F1-score (binary)
  - Confusion matrix
  - Per-class metrics
  - All scores saved to ml/validation/results/

Usage:
    cd backend
    python -m ml.validation.training.evaluate_validator

DO NOT run this until train_validator.py has completed and saved
the model to ml/validation/models/arecanut_validator.keras.
"""

import os
import sys
import json
import csv
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.dirname(SCRIPT_DIR)
ML_DIR        = os.path.dirname(VALIDATION_DIR)

PREPARED_DIR = os.path.join(VALIDATION_DIR, "prepared_data")
RESULTS_DIR  = os.path.join(VALIDATION_DIR, "results")
MODELS_DIR   = os.path.join(VALIDATION_DIR, "models")

TEST_CSV    = os.path.join(PREPARED_DIR, "manifest_test.csv")
MODEL_PATH  = os.path.join(MODELS_DIR, "arecanut_validator.keras")
INDEX_PATH  = os.path.join(MODELS_DIR, "validator_class_indices.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "validator_classification_report.txt")
METRICS_PATH= os.path.join(RESULTS_DIR, "validator_evaluation_metrics.json")
CM_PATH     = os.path.join(RESULTS_DIR, "validator_confusion_matrix.png")

IMG_SIZE = 224
BATCH_SIZE = 32
THRESHOLD = 0.5   # sigmoid decision threshold: >= 0.5 → arecanut

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["image_path"], row["class_label"]))
    return rows


def make_dataset(rows):
    """Build a tf.data pipeline for inference (no augmentation)."""
    paths  = [r[0] for r in rows]
    labels = [1 if r[1] == "arecanut" else 0 for r in rows]

    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.cond(
            tf.image.is_jpeg(path) if hasattr(tf.image, 'is_jpeg') else tf.constant(True),
            lambda: tf.io.decode_jpeg(img, channels=3),
            lambda: tf.io.decode_png(img, channels=3),
        )
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.cast(img, tf.float32)
        )
        return img, label

    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = ["not_arecanut", "arecanut"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=12, color=color)

    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title("Validator Confusion Matrix (normalised)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  ArecaNut Validator — Test Set Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 70)

    # ── Check model exists ──────────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        print(f"\n  [FATAL] Model not found: {MODEL_PATH}")
        print("  Run train_validator.py first.")
        sys.exit(1)

    if not os.path.isfile(TEST_CSV):
        print(f"\n  [FATAL] Test manifest not found: {TEST_CSV}")
        print("  Run prepare_split.py first.")
        sys.exit(1)

    # ── Load model and class mapping ────────────────────────────────────────
    print("\n  Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    with open(INDEX_PATH, encoding="utf-8") as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    print(f"  Classes: {idx_to_class}")

    # ── Load test data ──────────────────────────────────────────────────────
    test_rows = load_csv(TEST_CSV)
    print(f"  Test images: {len(test_rows)}")

    y_true = np.array([1 if r[1] == "arecanut" else 0 for r in test_rows])
    test_ds = make_dataset(test_rows)

    # ── Run predictions ──────────────────────────────────────────────────────
    print("  Running predictions...")
    preds_raw = model.predict(test_ds, verbose=0)
    preds_prob = preds_raw.flatten()  # sigmoid output [0, 1]
    y_pred = (preds_prob >= THRESHOLD).astype(int)

    # ── Compute metrics ─────────────────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

    # Classification report
    target_names = ["not_arecanut", "arecanut"]
    report_str = classification_report(y_true, y_pred,
                                       target_names=target_names,
                                       digits=4, zero_division=0)
    print("\n  Per-class report:")
    for line in report_str.splitlines():
        print(f"    {line}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion matrix:")
    print(f"    {cm}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, CM_PATH)
    print(f"\n  Saved: {CM_PATH}")

    # Save text report
    report_lines = [
        "=" * 70,
        "  ArecaNut Validator — Test Set Evaluation Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        "=" * 70,
        "",
        f"  Model: {MODEL_PATH}",
        f"  Threshold: {THRESHOLD}",
        f"  Test images: {len(test_rows)}",
        "",
        f"  Accuracy : {acc:.4f}",
        f"  Precision: {prec:.4f}",
        f"  Recall   : {rec:.4f}",
        f"  F1-score : {f1:.4f}",
        "",
        "  Per-class report:",
        report_str,
        "",
        f"  Confusion matrix:",
        f"    {cm}",
        "",
        "  Limitations:",
        "    - The validator distinguishes arecanut-like images from",
        "      clearly unrelated images. It may accept other palm species",
        "      or tropical plants as arecanut.",
        "    - Performance depends on the diversity of negative images.",
        "    - The test set should not be used to tune the model.",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {REPORT_PATH}")

    # Save JSON metrics
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "test_images": len(test_rows),
        "threshold": THRESHOLD,
        "confusion_matrix": cm.tolist(),
        "model_path": MODEL_PATH,
        "generated": datetime.now().isoformat(),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {METRICS_PATH}")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
