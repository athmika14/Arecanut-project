# -*- coding: utf-8 -*-
"""
train_mobilenetv2.py  —  ArecaNut AI: Disease Detection
========================================================
MobileNetV2 transfer-learning pipeline for 9-class arecanut disease
classification using the leakage-free source-group split manifests.

Environment:
  TensorFlow 2.21.0  |  Python 3.12  |  CPU-only (Windows native)

Phases:
  Phase 1 — Frozen backbone: train classification head only (≤20 epochs)
  Phase 2 — Fine-tune:       unfreeze top-30 MobileNetV2 layers  (≤10 epochs)

Outputs:
  ml/disease/models/  mobilenetv2_best.keras  class_indices.json
  ml/disease/results/ training_history.csv  accuracy_loss_plot.png
                      confusion_matrix.png  classification_report.txt
                      evaluation_metrics.json

Usage:
    python train_mobilenetv2.py

DO NOT modify dataset images or CSV manifests.
DO NOT retrain to optimise test metrics — test is evaluated once at the end.
"""

import os, sys, json, csv, time, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress oneDNN noise
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"   # suppress C++ INFO logs
warnings.filterwarnings("ignore", category=UserWarning)

import random
import numpy as np

# Fix seeds before importing TF
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ML_DISEASE   = os.path.dirname(SCRIPT_DIR)
PREPARED_DIR = os.path.join(ML_DISEASE, "prepared_data")
RESULTS_DIR  = os.path.join(ML_DISEASE, "results")
MODELS_DIR   = os.path.join(ML_DISEASE, "models")

TRAIN_CSV = os.path.join(PREPARED_DIR, "manifest_train.csv")
VAL_CSV   = os.path.join(PREPARED_DIR, "manifest_val.csv")
TEST_CSV  = os.path.join(PREPARED_DIR, "manifest_test.csv")

MODEL_PATH   = os.path.join(MODELS_DIR, "mobilenetv2_best.keras")
INDEX_PATH   = os.path.join(MODELS_DIR, "class_indices.json")
HISTORY_CSV  = os.path.join(RESULTS_DIR, "training_history.csv")
PLOT_PATH    = os.path.join(RESULTS_DIR, "accuracy_loss_plot.png")
CM_PATH      = os.path.join(RESULTS_DIR, "confusion_matrix.png")
REPORT_PATH  = os.path.join(RESULTS_DIR, "classification_report.txt")
METRICS_PATH = os.path.join(RESULTS_DIR, "evaluation_metrics.json")

for d in [MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
PHASE1_EPOCHS = 20      # max epochs for frozen-backbone phase
PHASE2_EPOCHS = 10      # max epochs for fine-tune phase
DROPOUT_RATE  = 0.3
DENSE_UNITS   = 128
FINETUNE_LAYERS = 30    # unfreeze this many layers from the top of MobileNetV2
LR_PHASE1     = 1e-3
LR_PHASE2     = 1e-4
ES_PATIENCE   = 5       # early stopping patience
RLROP_PATIENCE= 3       # ReduceLROnPlateau patience

EXPECTED_CLASSES = [
    "Healthy_Leaf", "Healthy_Nut", "Healthy_Trunk", "Mahali_Koleroga",
    "Stem_bleeding", "bud borer", "healthy_foot", "stem cracking",
    "yellow leaf disease",
]

# ── Utility ───────────────────────────────────────────────────────────────────

def sep(char="─", width=72):
    print(char * width)

def h(title):
    sep()
    print(f"  {title}")
    sep()

def load_csv(path):
    """Load a manifest CSV into lists of (image_path, class_label)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["image_path"], row["class_label"]))
    return rows

def validate_paths_and_labels(rows, split_name, valid_classes):
    """
    Check every path exists and every label is known.
    Returns (ok: bool, missing: list, bad_labels: list).
    """
    missing    = [p for p, _ in rows if not os.path.isfile(p)]
    bad_labels = [(p, l) for p, l in rows if l not in valid_classes]
    ok = (len(missing) == 0 and len(bad_labels) == 0)
    if missing:
        print(f"  [ERROR] {split_name}: {len(missing)} missing image path(s).")
        for p in missing[:5]:
            print(f"    {p}")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more.")
    if bad_labels:
        print(f"  [ERROR] {split_name}: {len(bad_labels)} unknown label(s).")
        for p, l in bad_labels[:5]:
            print(f"    label='{l}'  path={os.path.basename(p)}")
    return ok, missing, bad_labels

def make_dataset(rows, class_to_idx, training=False):
    """
    Build a tf.data pipeline from a list of (path, label) rows.
    """
    paths  = [r[0] for r in rows]
    labels = [class_to_idx[r[1]] for r in rows]

    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
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

def build_model(num_classes):
    """
    MobileNetV2 backbone (frozen) + custom classification head.
    Returns the full model with backbone frozen.
    """
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
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base

def save_history_csv(history_dict, path):
    """
    Write per-epoch training metrics to a CSV file.

    Separates per-epoch metric lists from scalar metadata values (e.g.
    ``_phase1_epochs``) so that only genuine sequence columns are written
    as CSV rows.  Scalar metadata is preserved as comment lines at the
    end of the file so no information is discarded.

    Bug fixed: the original version iterated over all keys including
    ``_phase1_epochs`` (an int) and attempted ``history_dict[k][i]``
    on it, raising ``TypeError: 'int' object is not subscriptable``.
    """
    if not history_dict:
        return

    # Split: per-epoch lists vs scalar metadata
    epoch_keys = [k for k, v in history_dict.items() if isinstance(v, list)]
    meta_keys  = [k for k, v in history_dict.items() if not isinstance(v, list)]

    if not epoch_keys:
        print("  [WARN] save_history_csv: no list-valued keys — nothing written.")
        return

    # Guard against accidental length mismatch between metric lists
    lengths  = {k: len(history_dict[k]) for k in epoch_keys}
    if len(set(lengths.values())) > 1:
        print(f"  [WARN] save_history_csv: unequal epoch counts {lengths}; "
              f"using minimum.")
    n_epochs = min(lengths.values())

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch"] + epoch_keys)
        w.writeheader()
        for i in range(n_epochs):
            row = {"epoch": i + 1}
            row.update({k: history_dict[k][i] for k in epoch_keys})
            w.writerow(row)
        # Preserve scalar metadata as comment lines (not CSV columns)
        for mk in meta_keys:
            f.write(f"# {mk}={history_dict[mk]}\n")

def plot_history(history_dict, save_path):
    """Save accuracy/loss curves for all epochs."""
    epochs = range(1, len(history_dict["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs, history_dict["accuracy"],     "b-o", label="Train", ms=4)
    axes[0].plot(epochs, history_dict["val_accuracy"], "r-o", label="Val",   ms=4)
    axes[0].set_title("Accuracy", fontsize=13)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    # Loss
    axes[1].plot(epochs, history_dict["loss"],     "b-o", label="Train", ms=4)
    axes[1].plot(epochs, history_dict["val_loss"], "r-o", label="Val",   ms=4)
    axes[1].set_title("Loss", fontsize=13)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Phase boundary line if fine-tuning happened
    p1_len = history_dict.get("_phase1_epochs", None)
    if p1_len:
        for ax in axes:
            ax.axvline(x=p1_len + 0.5, color="green", linestyle="--",
                       alpha=0.7, label="Fine-tune start")

    fig.suptitle("ArecaNut Disease Detection — MobileNetV2 Training", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Save normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), max(8, n * 1.0)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=7, color=color)

    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title("Confusion Matrix (normalised)\nArecaNut Disease Detection",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    start_wall = time.time()

    print()
    print("=" * 72)
    print("  ArecaNut AI — Disease Detection — MobileNetV2 Training")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 72)
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  Device     : CPU (GPU not available on Windows TF>=2.11)")
    print(f"  Seed       : {SEED}")
    print(f"  Image size : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size : {BATCH_SIZE}")
    print()

    # ── Step 1: Load and validate manifests ──────────────────────────────────
    h("STEP 1 — Load & Validate Manifests")
    print()

    for csv_path, name in [(TRAIN_CSV, "train"), (VAL_CSV, "val"), (TEST_CSV, "test")]:
        if not os.path.isfile(csv_path):
            print(f"  [FATAL] CSV not found: {csv_path}")
            sys.exit(1)

    train_rows = load_csv(TRAIN_CSV)
    val_rows   = load_csv(VAL_CSV)
    test_rows  = load_csv(TEST_CSV)

    print(f"  Rows loaded — train:{len(train_rows)}  val:{len(val_rows)}  test:{len(test_rows)}")
    print()

    # Build class index from training labels (sorted for determinism)
    train_labels_all = sorted({r[1] for r in train_rows})
    missing_expected = [c for c in EXPECTED_CLASSES if c not in train_labels_all]
    if missing_expected:
        print(f"  [FATAL] Expected classes missing from train CSV: {missing_expected}")
        sys.exit(1)

    class_to_idx = {cls: i for i, cls in enumerate(sorted(train_labels_all))}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    num_classes  = len(class_to_idx)
    print(f"  Classes ({num_classes}):")
    for cls, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"    {idx:>2}  {cls}")
    print()

    # Validate paths and labels
    all_ok = True
    for rows, name in [(train_rows, "TRAIN"), (val_rows, "VAL"), (test_rows, "TEST")]:
        ok, _, _ = validate_paths_and_labels(rows, name, set(class_to_idx.keys()))
        if not ok:
            all_ok = False
    if not all_ok:
        print()
        print("  [FATAL] Path or label validation failed. Fix before training.")
        sys.exit(1)
    print("  [OK] All image paths exist. All labels valid.")

    # ── Step 2: Class distribution ───────────────────────────────────────────
    h("STEP 2 — Class Distribution (Training Partition)")
    print()

    from collections import Counter
    train_label_list = [r[1] for r in train_rows]
    counts = Counter(train_label_list)
    max_count = max(counts.values())
    total_tr  = len(train_rows)

    for cls in sorted(counts.keys()):
        cnt = counts[cls]
        bar = "#" * max(1, round(cnt / max_count * 40))
        pct = cnt / total_tr * 100
        print(f"  {cls:<30}  {cnt:>5}  ({pct:4.1f}%)  {bar}")
    print()
    print(f"  Imbalance ratio: {max(counts.values())}:{min(counts.values())} "
          f"(~{round(max(counts.values())/min(counts.values()))}:1)")
    print()
    print("  [NOTE] healthy_foot: only 7 source groups — val/test have 1 source each.")
    print("         Per-class metrics for this class may not be statistically reliable.")

    # ── Step 3: Class weights ────────────────────────────────────────────────
    h("STEP 3 — Class Weights (from training partition only)")
    print()

    y_train_int = np.array([class_to_idx[r[1]] for r in train_rows])
    cw_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train_int,
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(cw_array)}
    for i, w in sorted(class_weight_dict.items()):
        print(f"  {i:>2}  {idx_to_class[i]:<30}  weight={w:.4f}")

    # ── Step 4: Build tf.data pipelines ─────────────────────────────────────
    h("STEP 4 — Building tf.data Pipelines")
    print()

    train_ds = make_dataset(train_rows, class_to_idx, training=True)
    val_ds   = make_dataset(val_rows,   class_to_idx, training=False)
    test_ds  = make_dataset(test_rows,  class_to_idx, training=False)

    print(f"  Train batches : {len(train_ds)}")
    print(f"  Val   batches : {len(val_ds)}")
    print(f"  Test  batches : {len(test_ds)}")

    # ── Step 5: Build model ──────────────────────────────────────────────────
    h("STEP 5 — Building Model")
    print()

    print("  Downloading / loading MobileNetV2 ImageNet weights ...")
    try:
        model, base_model = build_model(num_classes)
    except Exception as e:
        print(f"  [FATAL] Could not load MobileNetV2 weights: {e}")
        print("  Check network access to storage.googleapis.com and retry.")
        sys.exit(1)

    trainable_count = sum(
        np.prod(v.shape) for v in model.trainable_weights)
    total_count = sum(
        np.prod(v.shape) for v in model.weights)
    print(f"  Backbone        : MobileNetV2 (frozen)")
    print(f"  Head            : GAP -> Dense({DENSE_UNITS},relu) -> Dropout({DROPOUT_RATE}) -> Dense({num_classes},softmax)")
    print(f"  Total params    : {total_count:,}")
    print(f"  Trainable params: {trainable_count:,}  ({trainable_count/total_count*100:.1f}%)")
    print()

    # ── Step 6: Phase 1 — Train classification head (backbone frozen) ────────
    h("STEP 6 — Phase 1: Training Classification Head (Backbone Frozen)")
    print()
    print(f"  Max epochs : {PHASE1_EPOCHS}")
    print(f"  LR         : {LR_PHASE1}")
    print(f"  Early stop : patience={ES_PATIENCE} on val_accuracy")
    print()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PHASE1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=ES_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=RLROP_PATIENCE,
            min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]

    p1_start = time.time()
    hist_p1  = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks_p1,
        verbose=1,
    )
    p1_elapsed = time.time() - p1_start

    best_p1_val_acc  = max(hist_p1.history["val_accuracy"])
    best_p1_val_loss = min(hist_p1.history["val_loss"])
    actual_p1_epochs = len(hist_p1.history["accuracy"])

    print()
    print(f"  Phase 1 complete — {actual_p1_epochs} epochs in {p1_elapsed/60:.1f} min")
    print(f"  Best val_accuracy : {best_p1_val_acc:.4f}")
    print(f"  Best val_loss     : {best_p1_val_loss:.4f}")

    # ── Step 7: Phase 2 — Fine-tune upper backbone layers ───────────────────
    h("STEP 7 — Phase 2: Fine-Tuning Upper MobileNetV2 Layers")
    print()
    print(f"  Unfreezing top {FINETUNE_LAYERS} layers of MobileNetV2")
    print(f"  Max epochs : {PHASE2_EPOCHS}")
    print(f"  LR         : {LR_PHASE2}  (10x lower than Phase 1)")
    print()

    base_model.trainable = True
    # Freeze everything except the last FINETUNE_LAYERS layers
    for layer in base_model.layers[:-FINETUNE_LAYERS]:
        layer.trainable = False

    ft_trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
    print(f"  Trainable params after unfreeze: {ft_trainable:,}")
    print()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=ES_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=RLROP_PATIENCE,
            min_lr=1e-8, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]

    p2_start = time.time()
    hist_p2  = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks_p2,
        verbose=1,
    )
    p2_elapsed = time.time() - p2_start

    best_p2_val_acc  = max(hist_p2.history["val_accuracy"])
    actual_p2_epochs = len(hist_p2.history["accuracy"])

    print()
    print(f"  Phase 2 complete — {actual_p2_epochs} epochs in {p2_elapsed/60:.1f} min")
    print(f"  Best val_accuracy : {best_p2_val_acc:.4f}")

    # ── Step 8: Save combined history and plot ───────────────────────────────
    h("STEP 8 — Saving Training History & Plots")
    print()

    # Merge Phase 1 + Phase 2 history
    combined = {}
    for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        combined[key] = hist_p1.history[key] + hist_p2.history[key]
    combined["_phase1_epochs"] = actual_p1_epochs   # used by plotter

    save_history_csv(combined, HISTORY_CSV)
    print(f"  Saved: {HISTORY_CSV}")

    plot_history(combined, PLOT_PATH)

    # ── Step 9: Save class index mapping ────────────────────────────────────
    h("STEP 9 — Saving Class Index Mapping")
    print()

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"  Saved: {INDEX_PATH}")
    print(f"  Content: {idx_to_class}")

    # ── Step 10: Load best model and evaluate on TEST SET ────────────────────
    h("STEP 10 — Evaluating Best Model on TEST SET (one-time evaluation)")
    print()
    print("  Loading best checkpoint ...")

    best_model = tf.keras.models.load_model(MODEL_PATH)

    print("  Running predictions on test set ...")
    y_true = np.array([class_to_idx[r[1]] for r in test_rows])
    y_pred_probs = best_model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall accuracy
    test_accuracy = accuracy_score(y_true, y_pred)
    print()
    print(f"  Test accuracy : {test_accuracy:.4f}  ({test_accuracy*100:.2f}%)")

    # Classification report
    target_names = [idx_to_class[i] for i in range(num_classes)]
    report_str = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print()
    print("  Per-class classification report:")
    for line in report_str.split("\n"):
        print(f"    {line}")

    # Confusion matrix plot
    plot_confusion_matrix(y_true, y_pred, target_names, CM_PATH)

    # ── Step 11: Save classification report txt ──────────────────────────────
    h("STEP 11 — Saving Results")
    print()

    report_lines = []
    report_lines.append("=" * 72)
    report_lines.append("  ArecaNut AI — Disease Detection — Test Set Evaluation")
    report_lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    report_lines.append("=" * 72)
    report_lines.append("")
    report_lines.append("  Configuration")
    report_lines.append(f"    Model         : MobileNetV2 + custom head")
    report_lines.append(f"    Input size    : {IMG_SIZE}x{IMG_SIZE} RGB")
    report_lines.append(f"    Batch size    : {BATCH_SIZE}")
    report_lines.append(f"    Phase 1 epochs trained : {actual_p1_epochs}")
    report_lines.append(f"    Phase 2 epochs trained : {actual_p2_epochs}")
    report_lines.append(f"    Best val accuracy (P1) : {best_p1_val_acc:.4f}")
    report_lines.append(f"    Best val accuracy (P2) : {best_p2_val_acc:.4f}")
    report_lines.append(f"    Random seed   : {SEED}")
    report_lines.append(f"    Device        : CPU (Windows native TF>=2.11)")
    report_lines.append("")
    report_lines.append("  Dataset")
    report_lines.append(f"    Train images  : {len(train_rows)}")
    report_lines.append(f"    Val images    : {len(val_rows)}")
    report_lines.append(f"    Test images   : {len(test_rows)}")
    report_lines.append(f"    Split         : leakage-free source-group stratified")
    report_lines.append("")
    report_lines.append("  Test Set Results")
    report_lines.append(f"    Accuracy      : {test_accuracy:.4f}")
    report_lines.append("")
    report_lines.append("  Per-class report (test set):")
    report_lines.append(report_str)
    report_lines.append("")
    report_lines.append("  Limitations")
    report_lines.append("    - CPU-only training: slower convergence, no GPU acceleration.")
    report_lines.append("    - healthy_foot: only 7 independent source groups total.")
    report_lines.append("      Val and test each have 1 source (~10 images).")
    report_lines.append("      Per-class metrics for this class are not reliable.")
    report_lines.append("    - Class imbalance ~51:1; mitigated with class_weight.")
    report_lines.append("    - All images are augmented copies of 1,226 originals.")
    report_lines.append("      Source-group split prevents leakage but cannot account")
    report_lines.append("      for potential visual similarity between different sources")
    report_lines.append("      from the same field/camera session.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {REPORT_PATH}")

    # Save JSON metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics_dict = {
        "test_accuracy": float(test_accuracy),
        "macro_precision": float(precision_score(y_true, y_pred,
                                                  average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_true, y_pred,
                                              average="macro", zero_division=0)),
        "macro_f1":        float(f1_score(y_true, y_pred,
                                          average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred,
                                                     average="weighted", zero_division=0)),
        "weighted_recall":    float(recall_score(y_true, y_pred,
                                                  average="weighted", zero_division=0)),
        "weighted_f1":        float(f1_score(y_true, y_pred,
                                              average="weighted", zero_division=0)),
        "phase1_epochs_trained": actual_p1_epochs,
        "phase2_epochs_trained": actual_p2_epochs,
        "best_val_accuracy_p1":  float(best_p1_val_acc),
        "best_val_accuracy_p2":  float(best_p2_val_acc),
        "train_images":  len(train_rows),
        "val_images":    len(val_rows),
        "test_images":   len(test_rows),
        "seed":          SEED,
        "model_path":    MODEL_PATH,
        "generated":     datetime.now().isoformat(),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Saved: {METRICS_PATH}")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - start_wall
    print()
    print("=" * 72)
    print("  TRAINING COMPLETE")
    print("=" * 72)
    print(f"  Total wall time     : {total_elapsed/60:.1f} min")
    print(f"  Phase 1 epochs      : {actual_p1_epochs} / {PHASE1_EPOCHS} max")
    print(f"  Phase 2 epochs      : {actual_p2_epochs} / {PHASE2_EPOCHS} max")
    print(f"  Best val acc (P1)   : {best_p1_val_acc:.4f}")
    print(f"  Best val acc (P2)   : {best_p2_val_acc:.4f}")
    print(f"  Test accuracy       : {test_accuracy:.4f}  ({test_accuracy*100:.2f}%)")
    print(f"  Macro F1            : {metrics_dict['macro_f1']:.4f}")
    print(f"  Weighted F1         : {metrics_dict['weighted_f1']:.4f}")
    print()
    print("  Output files:")
    for path in [MODEL_PATH, INDEX_PATH, HISTORY_CSV,
                 PLOT_PATH, CM_PATH, REPORT_PATH, METRICS_PATH]:
        exists = "[OK]" if os.path.isfile(path) else "[MISSING]"
        print(f"    {exists}  {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
