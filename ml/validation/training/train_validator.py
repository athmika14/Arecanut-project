# -*- coding: utf-8 -*-
"""
train_validator.py — ArecaNut Validator: Binary Classifier Training
=====================================================================
Trains a MobileNetV2-based binary classifier to distinguish arecanut
images from non-arecanut images.

The model is saved to:
    ml/validation/models/arecanut_validator.keras

Class mapping is saved to:
    ml/validation/models/validator_class_indices.json
        {"0": "not_arecanut", "1": "arecanut"}

Usage:
    cd backend
    python -m ml.validation.training.train_validator

DO NOT run this until:
  1. audit_dataset.py reports sufficient negative images.
  2. prepare_split.py has created manifests in ml/validation/prepared_data/.
"""

import os
import sys
import json
import csv
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
warnings.filterwarnings("ignore")

import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.dirname(SCRIPT_DIR)
ML_DIR        = os.path.dirname(VALIDATION_DIR)
PROJECT_DIR   = os.path.dirname(ML_DIR)

PREPARED_DIR = os.path.join(VALIDATION_DIR, "prepared_data")
RESULTS_DIR  = os.path.join(VALIDATION_DIR, "results")
MODELS_DIR   = os.path.join(VALIDATION_DIR, "models")

TRAIN_CSV = os.path.join(PREPARED_DIR, "manifest_train.csv")
VAL_CSV   = os.path.join(PREPARED_DIR, "manifest_val.csv")
TEST_CSV  = os.path.join(PREPARED_DIR, "manifest_test.csv")

MODEL_PATH    = os.path.join(MODELS_DIR, "arecanut_validator.keras")
INDEX_PATH    = os.path.join(MODELS_DIR, "validator_class_indices.json")
HISTORY_CSV   = os.path.join(RESULTS_DIR, "validator_training_history.csv")
PLOT_PATH     = os.path.join(RESULTS_DIR, "validator_accuracy_loss_plot.png")

for d in [MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────

IMG_SIZE       = 224
BATCH_SIZE     = 32
PHASE1_EPOCHS  = 20
PHASE2_EPOCHS  = 10
DROPOUT_RATE   = 0.3
DENSE_UNITS    = 128
FINETUNE_LAYERS = 30
LR_PHASE1      = 1e-3
LR_PHASE2      = 1e-4
ES_PATIENCE    = 5
RLROP_PATIENCE = 3

# Class mapping: 0 = not_arecanut, 1 = arecanut
CLASS_TO_IDX = {"not_arecanut": 0, "arecanut": 1}
IDX_TO_CLASS = {0: "not_arecanut", 1: "arecanut"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["image_path"], row["class_label"], row["source_id"]))
    return rows


def make_dataset(rows, training=False):
    """Build a tf.data pipeline from (path, label) rows."""
    paths  = [r[0] for r in rows]
    labels = [CLASS_TO_IDX[r[1]] for r in rows]

    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        # Try JPEG first, fall back to PNG
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

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=min(len(rows), 2000), seed=SEED)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def build_model():
    """MobileNetV2 backbone (frozen) + binary classification head."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    # Single sigmoid output: 1 = arecanut, 0 = not_arecanut
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def save_history_csv(history_dict, path):
    """Write epoch history to CSV (handles scalar metadata)."""
    if not history_dict:
        return
    epoch_keys = [k for k, v in history_dict.items() if isinstance(v, list)]
    meta_keys  = [k for k, v in history_dict.items() if not isinstance(v, list)]
    if not epoch_keys:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        import csv as csv_mod
        w = csv_mod.DictWriter(f, fieldnames=["epoch"] + epoch_keys)
        w.writeheader()
        n = len(history_dict[epoch_keys[0]])
        for i in range(n):
            row = {"epoch": i + 1}
            row.update({k: history_dict[k][i] for k in epoch_keys})
            w.writerow(row)
        for mk in meta_keys:
            f.write(f"# {mk}={history_dict[mk]}\n")


def plot_history(history_dict, save_path):
    """Save accuracy/loss curves."""
    epochs = range(1, len(history_dict["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history_dict["accuracy"],     "b-o", label="Train", ms=4)
    axes[0].plot(epochs, history_dict["val_accuracy"], "r-o", label="Val",   ms=4)
    axes[0].set_title("Accuracy", fontsize=13)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history_dict["loss"],     "b-o", label="Train", ms=4)
    axes[1].plot(epochs, history_dict["val_loss"], "r-o", label="Val",   ms=4)
    axes[1].set_title("Loss", fontsize=13)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    p1_len = history_dict.get("_phase1_epochs", None)
    if p1_len:
        for ax in axes:
            ax.axvline(x=p1_len + 0.5, color="green", linestyle="--",
                       alpha=0.7, label="Fine-tune start")

    fig.suptitle("ArecaNut Validator — MobileNetV2 Binary Training", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    start_wall = time.time()

    print("\n" + "=" * 70)
    print("  ArecaNut Validator — Binary Classifier Training")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 70)
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  Seed       : {SEED}")
    print(f"  Image size : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size : {BATCH_SIZE}")
    print()

    # ── Step 1: Load manifests ───────────────────────────────────────────────
    print("── Step 1: Loading manifests ──")
    for p, name in [(TRAIN_CSV, "train"), (VAL_CSV, "val"), (TEST_CSV, "test")]:
        if not os.path.isfile(p):
            print(f"  [FATAL] {name} manifest not found: {p}")
            print("  Run prepare_split.py first.")
            sys.exit(1)

    train_rows = load_csv(TRAIN_CSV)
    val_rows   = load_csv(VAL_CSV)
    test_rows  = load_csv(TEST_CSV)
    print(f"  Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)}")

    # ── Step 2: Class distribution ───────────────────────────────────────────
    print("\n── Step 2: Class distribution (train) ──")
    train_labels = [r[1] for r in train_rows]
    counts = Counter(train_labels)
    for cls in sorted(counts.keys()):
        n = counts[cls]
        print(f"  {cls:<20}  {n:>5}  ({n/len(train_rows)*100:.1f}%)")

    # ── Step 3: Class weights ────────────────────────────────────────────────
    print("\n── Step 3: Class weights ──")
    y_int = np.array([CLASS_TO_IDX[l] for l in train_labels])
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_int)
    class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}
    for i in [0, 1]:
        print(f"  {IDX_TO_CLASS[i]:<20}  weight={class_weight_dict[i]:.4f}")

    # ── Step 4: Build datasets ───────────────────────────────────────────────
    print("\n── Step 4: Building tf.data pipelines ──")
    train_ds = make_dataset(train_rows, training=True)
    val_ds   = make_dataset(val_rows,   training=False)
    print(f"  Train batches: {len(train_ds)}")
    print(f"  Val   batches: {len(val_ds)}")

    # ── Step 5: Build model ──────────────────────────────────────────────────
    print("\n── Step 5: Building model ──")
    model, base_model = build_model()
    tp = sum(int(tf.size(v)) for v in model.trainable_weights)
    tot = sum(int(tf.size(v)) for v in model.weights)
    print(f"  Total params: {tot:,}  Trainable: {tp:,}  ({tp/tot*100:.1f}%)")

    # ── Step 6: Phase 1 — Frozen backbone ───────────────────────────────────
    print("\n── Step 6: Phase 1 — Frozen backbone ──")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PHASE1),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_p1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=ES_PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=RLROP_PATIENCE,
            min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1),
    ]
    hist_p1 = model.fit(train_ds, validation_data=val_ds,
                        epochs=PHASE1_EPOCHS, class_weight=class_weight_dict,
                        callbacks=callbacks_p1, verbose=1)

    best_p1_val_acc = max(hist_p1.history["val_accuracy"])
    actual_p1 = len(hist_p1.history["accuracy"])
    print(f"\n  Phase 1: {actual_p1} epochs, best val_acc={best_p1_val_acc:.4f}")

    # ── Step 7: Phase 2 — Fine-tune ──────────────────────────────────────────
    print("\n── Step 7: Phase 2 — Fine-tuning top layers ──")
    base_model.trainable = True
    for layer in base_model.layers[:-FINETUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_p2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=ES_PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=RLROP_PATIENCE,
            min_lr=1e-8, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1),
    ]
    hist_p2 = model.fit(train_ds, validation_data=val_ds,
                        epochs=PHASE2_EPOCHS, class_weight=class_weight_dict,
                        callbacks=callbacks_p2, verbose=1)
    best_p2_val_acc = max(hist_p2.history["val_accuracy"])
    actual_p2 = len(hist_p2.history["accuracy"])
    print(f"\n  Phase 2: {actual_p2} epochs, best val_acc={best_p2_val_acc:.4f}")

    # ── Step 8: Save history and plots ───────────────────────────────────────
    print("\n── Step 8: Saving history and plots ──")
    combined = {}
    for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        combined[key] = hist_p1.history[key] + hist_p2.history[key]
    combined["_phase1_epochs"] = actual_p1

    save_history_csv(combined, HISTORY_CSV)
    print(f"  Saved: {HISTORY_CSV}")
    plot_history(combined, PLOT_PATH)
    print(f"  Saved: {PLOT_PATH}")

    # ── Step 9: Save class mapping ───────────────────────────────────────────
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(IDX_TO_CLASS, f, indent=2)
    print(f"  Saved: {INDEX_PATH}")

    # ── Summary ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - start_wall
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Phase 1: {actual_p1} epochs  (best val_acc={best_p1_val_acc:.4f})")
    print(f"  Phase 2: {actual_p2} epochs  (best val_acc={best_p2_val_acc:.4f})")
    print(f"  Wall time: {total_elapsed/60:.1f} min")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Class mapping: {INDEX_PATH}")
    print("\n  Next step: run evaluate_validator.py to evaluate on the test set.")
    print("=" * 70)


if __name__ == "__main__":
    main()
