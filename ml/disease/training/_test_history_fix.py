# -*- coding: utf-8 -*-
"""
Lightweight validation of save_history_csv — no training, no TF import.
Reproduces the exact bug and confirms the fix works.
"""
import csv
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "results")
TEST_CSV = os.path.join(RESULTS_DIR, "_test_history_validation.csv")

# ─── ORIGINAL (buggy) version ─────────────────────────────────────────────────
def save_history_csv_original(history_dict, path):
    if not history_dict:
        return
    keys = list(history_dict.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch"] + keys)
        w.writeheader()
        n_epochs = len(history_dict[keys[0]])
        for i in range(n_epochs):
            row = {"epoch": i + 1}
            row.update({k: history_dict[k][i] for k in keys})   # <-- bug
            w.writerow(row)


# ─── FIXED version ────────────────────────────────────────────────────────────
def save_history_csv_fixed(history_dict, path):
    """
    Write per-epoch metrics to a CSV.

    Separates scalar metadata (anything that is NOT a list) from
    per-epoch metric lists so that only genuine sequence columns are
    iterated over. Scalar metadata is recorded in a separate metadata
    section appended as comment rows at the bottom of the file.
    """
    if not history_dict:
        return

    # Split keys into per-epoch lists vs scalar metadata
    epoch_keys = [k for k, v in history_dict.items() if isinstance(v, list)]
    meta_keys  = [k for k, v in history_dict.items() if not isinstance(v, list)]

    if not epoch_keys:
        print("  [WARN] save_history_csv: no list-valued keys found — nothing to write.")
        return

    # Validate that all list columns have the same length
    lengths = {k: len(history_dict[k]) for k in epoch_keys}
    if len(set(lengths.values())) > 1:
        print(f"  [WARN] save_history_csv: inconsistent epoch counts: {lengths}. "
              f"Using minimum length.")
    n_epochs = min(lengths.values())

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch"] + epoch_keys)
        w.writeheader()
        for i in range(n_epochs):
            row = {"epoch": i + 1}
            row.update({k: history_dict[k][i] for k in epoch_keys})
            w.writerow(row)
        # Append scalar metadata as comment rows so no information is lost
        for mk in meta_keys:
            f.write(f"# {mk}={history_dict[mk]}\n")


# ─── Build a combined dict exactly as Step 8 does ────────────────────────────
fake_p1 = {
    "accuracy":     [0.70, 0.80, 0.85],
    "val_accuracy": [0.65, 0.75, 0.78],
    "loss":         [1.00, 0.80, 0.65],
    "val_loss":     [1.10, 0.90, 0.72],
}
fake_p2 = {
    "accuracy":     [0.88, 0.90],
    "val_accuracy": [0.82, 0.83],
    "loss":         [0.50, 0.45],
    "val_loss":     [0.60, 0.58],
}
actual_p1_epochs = len(fake_p1["accuracy"])   # 3

combined = {}
for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
    combined[key] = fake_p1[key] + fake_p2[key]
combined["_phase1_epochs"] = actual_p1_epochs  # integer — NOT a list

print("=" * 60)
print("  save_history_csv — Bug Reproduction & Fix Validation")
print("=" * 60)
print()
print(f"  combined keys          : {list(combined.keys())}")
print(f"  combined['accuracy']   : {combined['accuracy']}")
print(f"  combined['_phase1_epochs'] : {combined['_phase1_epochs']}  "
      f"(type={type(combined['_phase1_epochs']).__name__})")
print()

# ── Test 1: reproduce the original error ─────────────────────────────────────
print("── Test 1: Original function (expect TypeError) ──")
try:
    save_history_csv_original(combined, TEST_CSV)
    print("  UNEXPECTED: no error raised — bug may have changed!")
    sys.exit(1)
except TypeError as e:
    print(f"  [CONFIRMED BUG] TypeError: {e}")
print()

# ── Test 2: fixed function succeeds ──────────────────────────────────────────
print("── Test 2: Fixed function ──")
try:
    save_history_csv_fixed(combined, TEST_CSV)
    print(f"  [OK] CSV written to: {TEST_CSV}")
except Exception as e:
    print(f"  [FAIL] Unexpected error: {e}")
    sys.exit(1)

# ── Test 3: verify CSV contents ──────────────────────────────────────────────
print()
print("── Test 3: Verify CSV contents ──")
with open(TEST_CSV, "r", encoding="utf-8") as f:
    content = f.read()

print("  Raw file content:")
for line in content.splitlines():
    print(f"    {line}")

# Parse the epoch rows
rows = []
with open(TEST_CSV, "r", encoding="utf-8") as f:
    for row in csv.DictReader(line for line in f if not line.startswith("#")):
        rows.append(row)

expected_epochs = len(combined["accuracy"])   # 5 total
assert len(rows) == expected_epochs, f"Expected {expected_epochs} rows, got {len(rows)}"
print()
print(f"  Epoch rows: {len(rows)} (expected {expected_epochs}) — [OK]")

# Check that _phase1_epochs does NOT appear as a CSV column
assert "_phase1_epochs" not in rows[0], "_phase1_epochs should not be a CSV column"
print(f"  _phase1_epochs is NOT a CSV column — [OK]")

# Check the metadata comment line is present
assert "# _phase1_epochs=3" in content, "Metadata comment line missing"
print(f"  Metadata comment '# _phase1_epochs=3' present — [OK]")

# Check first and last row values
assert float(rows[0]["accuracy"]) == 0.70, f"Row 0 accuracy mismatch: {rows[0]['accuracy']}"
assert float(rows[-1]["val_loss"]) == 0.58, f"Last row val_loss mismatch: {rows[-1]['val_loss']}"
print(f"  First row accuracy=0.70, last row val_loss=0.58 — [OK]")

print()
print("── Test 4: Edge cases ──")

# Empty dict
save_history_csv_fixed({}, TEST_CSV + ".empty")
print("  Empty dict handled without crash — [OK]")

# Only scalar keys
save_history_csv_fixed({"_meta": 42}, TEST_CSV + ".scalar")
print("  All-scalar dict handled without crash — [OK]")

# Mismatched lengths (should warn but not crash)
save_history_csv_fixed(
    {"accuracy": [0.8, 0.9], "val_accuracy": [0.7]},
    TEST_CSV + ".mismatch"
)
print("  Mismatched-length lists handled without crash — [OK]")

print()
print("=" * 60)
print("  ALL TESTS PASSED")
print("=" * 60)

# Cleanup temp files
for f in [TEST_CSV, TEST_CSV+".empty", TEST_CSV+".scalar", TEST_CSV+".mismatch"]:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass
