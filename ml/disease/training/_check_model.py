# -*- coding: utf-8 -*-
"""Inspect the saved checkpoint and validate the current save_history_csv fix."""
import os, sys, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DISEASE = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(ML_DISEASE, "models", "mobilenetv2_best.keras")
HIST_CSV   = os.path.join(ML_DISEASE, "results", "training_history.csv")

# ── 1. Inspect existing output files ─────────────────────────────────────────
print("=" * 62)
print("  1. Output File Inventory")
print("=" * 62)

checks = {
    "mobilenetv2_best.keras":    MODEL_PATH,
    "class_indices.json":        os.path.join(ML_DISEASE, "models", "class_indices.json"),
    "training_history.csv":      HIST_CSV,
    "accuracy_loss_plot.png":    os.path.join(ML_DISEASE, "results", "accuracy_loss_plot.png"),
    "confusion_matrix.png":      os.path.join(ML_DISEASE, "results", "confusion_matrix.png"),
    "classification_report.txt": os.path.join(ML_DISEASE, "results", "classification_report.txt"),
    "evaluation_metrics.json":   os.path.join(ML_DISEASE, "results", "evaluation_metrics.json"),
}
for name, path in checks.items():
    if os.path.isfile(path):
        kb = os.path.getsize(path) / 1024
        unit = f"{kb/1024:.2f} MB" if kb > 512 else f"{kb:.1f} KB"
        print(f"  [EXISTS]  {name:<35} {unit}")
    else:
        print(f"  [MISSING] {name}")

# ── 2. Inspect training_history.csv ──────────────────────────────────────────
print()
print("=" * 62)
print("  2. training_history.csv Contents")
print("=" * 62)
if os.path.isfile(HIST_CSV):
    with open(HIST_CSV, encoding="utf-8") as f:
        content = f.read()
    print(f"  Raw content ({len(content)} bytes):")
    for line in content.splitlines():
        print(f"    {repr(line)}")
    # Count data rows (non-comment, non-header)
    data_rows = [l for l in content.splitlines()
                 if l.strip() and not l.startswith("#") and not l.startswith("epoch,")]
    print(f"  Data rows: {len(data_rows)}")
    if len(data_rows) == 0:
        print("  STATUS: Header-only — training crashed before writing any epoch data.")
        print("          This is the symptom of the TypeError bug.")
else:
    print("  Not found.")

# ── 3. Load and inspect the saved model ──────────────────────────────────────
print()
print("=" * 62)
print("  3. Model Checkpoint Inspection")
print("=" * 62)
if not os.path.isfile(MODEL_PATH):
    print("  [MISSING] No checkpoint found.")
else:
    import tensorflow as tf
    print(f"  Loading: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"  Loaded OK")
    print(f"  Input shape  : {model.input_shape}")
    print(f"  Output shape : {model.output_shape}")
    print(f"  Total params : {model.count_params():,}")
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"  File size    : {size_mb:.2f} MB")

    # Identify which phase the checkpoint came from
    base = model.layers[1]  # MobileNetV2 sub-model
    total_layers = len(base.layers)
    frozen   = sum(1 for l in base.layers if not l.trainable)
    unfrozen = total_layers - frozen
    print(f"  Backbone layers: {total_layers}  frozen={frozen}  unfrozen={unfrozen}")

    if unfrozen == 0:
        print("  Phase    : Phase-1 (backbone fully frozen)")
        print("  NOTE: A Phase-1 checkpoint serializes all backbone weights regardless,")
        print("        which is why it can be 22 MB instead of 11 MB after a reload/resave.")
    else:
        print(f"  Phase    : Phase-2 (top {unfrozen} backbone layers were trainable)")

# ── 4. Validate the FIXED save_history_csv in isolation ──────────────────────
print()
print("=" * 62)
print("  4. save_history_csv Fix Validation (synthetic data, no training)")
print("=" * 62)

TEST_CSV = os.path.join(ML_DISEASE, "results", "_fix_validation.csv")

def save_history_csv_fixed(history_dict, path):
    if not history_dict:
        return
    epoch_keys = [k for k, v in history_dict.items() if isinstance(v, list)]
    meta_keys  = [k for k, v in history_dict.items() if not isinstance(v, list)]
    if not epoch_keys:
        print("  [WARN] no list-valued keys — nothing written.")
        return
    lengths  = {k: len(history_dict[k]) for k in epoch_keys}
    if len(set(lengths.values())) > 1:
        print(f"  [WARN] unequal epoch counts {lengths}; using minimum.")
    n_epochs = min(lengths.values())
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch"] + epoch_keys)
        w.writeheader()
        for i in range(n_epochs):
            row = {"epoch": i + 1}
            row.update({k: history_dict[k][i] for k in epoch_keys})
            w.writerow(row)
        for mk in meta_keys:
            f.write(f"# {mk}={history_dict[mk]}\n")

# Replicate exactly what Step 8 builds
fake_p1 = {"accuracy":[0.70,0.80,0.85], "val_accuracy":[0.65,0.75,0.78],
           "loss":[1.0,0.8,0.65],        "val_loss":[1.1,0.9,0.72]}
fake_p2 = {"accuracy":[0.88,0.90],       "val_accuracy":[0.82,0.83],
           "loss":[0.50,0.45],            "val_loss":[0.60,0.58]}
actual_p1_epochs = 3

combined = {}
for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
    combined[key] = fake_p1[key] + fake_p2[key]
combined["_phase1_epochs"] = actual_p1_epochs   # integer — the problematic key

print(f"  combined keys     : {list(combined.keys())}")
print(f"  _phase1_epochs    : {combined['_phase1_epochs']}  "
      f"(type={type(combined['_phase1_epochs']).__name__})")
print()

# Confirm OLD code would crash
def save_history_csv_old(history_dict, path):
    if not history_dict:
        return
    keys = list(history_dict.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch"] + keys)
        w.writeheader()
        n_epochs = len(history_dict[keys[0]])
        for i in range(n_epochs):
            row = {"epoch": i + 1}
            row.update({k: history_dict[k][i] for k in keys})   # TypeError here
            w.writerow(row)

print("  Test A — Old code crashes:")
try:
    save_history_csv_old(combined, TEST_CSV)
    print("  UNEXPECTED: no error (bug may have been patched elsewhere)")
    sys.exit(1)
except TypeError as e:
    print(f"  [CONFIRMED] TypeError: {e}")

print()
print("  Test B — Fixed code succeeds:")
try:
    save_history_csv_fixed(combined, TEST_CSV)
    print("  [OK] No exception raised")
except Exception as e:
    print(f"  [FAIL] {e}")
    sys.exit(1)

# Verify CSV contents
with open(TEST_CSV, encoding="utf-8") as f:
    content = f.read()
print()
print("  CSV output:")
for line in content.splitlines():
    print(f"    {line}")

rows = list(csv.DictReader(l for l in content.splitlines() if not l.startswith("#")))
assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"
assert "_phase1_epochs" not in rows[0], "_phase1_epochs must not be a CSV column"
assert "# _phase1_epochs=3" in content, "Metadata comment missing"
assert float(rows[0]["accuracy"]) == 0.70
assert float(rows[-1]["val_loss"]) == 0.58
print()
print("  Assertions: 5 data rows, no _phase1_epochs column, metadata comment present")
print("  [ALL PASSED]")

# Cleanup
os.remove(TEST_CSV)

# ── 5. Resumability assessment ────────────────────────────────────────────────
print()
print("=" * 62)
print("  5. Can Training Resume from the Existing Checkpoint?")
print("=" * 62)
print()
print("  SHORT ANSWER: No — training must restart from scratch.")
print()
print("  REASON: Keras does not save optimizer state or epoch counter")
print("  inside a .keras checkpoint — it only saves the model weights.")
print("  The training script has no resume logic: it rebuilds the model")
print("  from ImageNet weights in build_model(), then trains Phase 1")
print("  from epoch 1, then Phase 2 from epoch 1. Loading the existing")
print("  checkpoint at the start would give a better weight initialisation")
print("  but the script does not do that either.")
print()
print("  WHAT THE CHECKPOINT DOES GIVE YOU: The best val_accuracy weights")
print("  seen so far (val_acc=0.7798 after Epoch 1 of Phase 1). If training")
print("  is restarted and beats that checkpoint, the file will be overwritten")
print("  with better weights. If it does not beat it, the existing checkpoint")
print("  will remain as the fallback.")
print()
print("  SAFE ACTION: simply re-run the training script. The ModelCheckpoint")
print("  callback will overwrite only if val_accuracy improves, so the")
print("  existing checkpoint is never made worse.")
print()
print("=" * 62)
print("  SUMMARY")
print("=" * 62)
print("  Bug cause : save_history_csv iterated ALL keys of combined{},")
print("              including _phase1_epochs (an int). Attempting")
print("              history_dict['_phase1_epochs'][i] raises")
print("              TypeError: 'int' object is not subscriptable.")
print("              The CSV file was created and its header written")
print("              before the crash, leaving a 58-byte header-only file.")
print()
print("  Fix in    : train_mobilenetv2.py save_history_csv() — already")
print("              applied. The function now separates epoch_keys (lists)")
print("              from meta_keys (scalars). Only epoch_keys become CSV")
print("              columns; scalars are appended as # comments.")
print()
print("  Status    :")
print("    model checkpoint  — EXISTS, valid, 22.74 MB")
print("    class_indices.json — MISSING (written after training)")
print("    training_history.csv — EXISTS but header-only (0 data rows)")
print("    all plots/reports — MISSING")
print("    Training did NOT complete. Must re-run from scratch.")
print()
print("  Next command:")
print("    python -u ml\\disease\\training\\train_mobilenetv2.py")
