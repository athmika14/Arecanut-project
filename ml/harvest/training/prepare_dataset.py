"""
prepare_dataset.py — ArecaNut AI: Harvest Readiness
=====================================================
Splits the original dataset into train / validation / test sets
and copies images into the prepared/ folder structure.

Original dataset (READ-ONLY, never modified):
    ml/harvest/dataset/DU_UBDTCE_Database/
        Ripeness_Arecanut_Bunches/Input_Images/   → ripe
        Unripen_Arecanut_Bunches/Input_Images/    → unripe

Output structure:
    ml/harvest/prepared/
        train/      ripe/  unripe/
        validation/ ripe/  unripe/
        test/       ripe/  unripe/

Split ratio  : 70% train | 15% validation | 15% test
Random seed  : 42  (fixed — reproducible every run)
Stratified   : yes — each split gets the same class proportions

Run:
    python prepare_dataset.py
Safe to re-run: files already present in destination are skipped.
"""

import os
import shutil
import random

# ── Configuration ────────────────────────────────────────────────────────────

# Locate this script's directory, then navigate from there.
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ML_HARVEST   = os.path.dirname(SCRIPT_DIR)                  # ml/harvest/

DATASET_ROOT = os.path.join(
    ML_HARVEST, "dataset", "DU_UBDTCE_Database"
)

# Map: source subfolder name  →  destination class name
CLASS_MAP = {
    "Ripeness_Arecanut_Bunches":  "ripe",
    "Unripen_Arecanut_Bunches":   "unripe",
}

# Only images inside Input_Images/ are used for training.
# Ground_Truth_Images/ (segmentation masks) are intentionally excluded.
INPUT_SUBFOLDER = "Input_Images"

PREPARED_ROOT = os.path.join(ML_HARVEST, "prepared")

SPLITS = ["train", "validation", "test"]

# 70 / 15 / 15
TRAIN_RATIO      = 0.70
VALIDATION_RATIO = 0.15
# test ratio is whatever remains (≈ 0.15)

RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_valid_image(filename: str) -> bool:
    """Return True if the file has an accepted image extension."""
    _, ext = os.path.splitext(filename)
    return ext.lower() in VALID_EXTENSIONS


def split_list(items: list, train_r: float, val_r: float, seed: int):
    """
    Shuffle and split a list into (train, val, test) sub-lists.
    Uses a fixed random seed so the result is reproducible.
    """
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = round(n * train_r)
    n_val   = round(n * val_r)
    # test gets the rest — avoids rounding losing/gaining images
    n_test  = n - n_train - n_val

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]

    assert len(test) == n_test
    return train, val, test


def copy_files(file_paths: list, dest_dir: str) -> tuple[int, int]:
    """
    Copy files to dest_dir.
    Returns (copied_count, skipped_count).
    Skipped means the file already exists — safe for re-runs.
    """
    os.makedirs(dest_dir, exist_ok=True)
    copied  = 0
    skipped = 0
    for src in file_paths:
        filename = os.path.basename(src)
        dst = os.path.join(dest_dir, filename)
        if os.path.exists(dst):
            skipped += 1
        else:
            shutil.copy2(src, dst)
            copied += 1
    return copied, skipped


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ArecaNut AI — Harvest Readiness Dataset Preparation")
    print("=" * 60)
    print(f"\nDataset root : {DATASET_ROOT}")
    print(f"Output root  : {PREPARED_ROOT}")
    print(f"Random seed  : {RANDOM_SEED}")
    print(f"Split ratio  : {int(TRAIN_RATIO*100)}% train  |  "
          f"{int(VALIDATION_RATIO*100)}% val  |  "
          f"{int((1 - TRAIN_RATIO - VALIDATION_RATIO)*100)}% test")
    print()

    # ── Step 1: Discover images ──────────────────────────────────────────────
    print("── Step 1: Reading original dataset ──")

    class_files   = {}   # { 'ripe': [path, ...], 'unripe': [...] }
    invalid_files = {}   # { 'ripe': [path, ...], 'unripe': [...] }

    for src_folder, class_label in CLASS_MAP.items():
        src_path = os.path.join(DATASET_ROOT, src_folder, INPUT_SUBFOLDER)

        if not os.path.isdir(src_path):
            print(f"  [ERROR] Folder not found: {src_path}")
            print("  Aborting.")
            return

        valid   = []
        invalid = []

        for fname in sorted(os.listdir(src_path)):
            full_path = os.path.join(src_path, fname)
            if os.path.isfile(full_path):
                if is_valid_image(fname):
                    valid.append(full_path)
                else:
                    invalid.append(full_path)

        class_files[class_label]   = valid
        invalid_files[class_label] = invalid

        print(f"  {class_label:<8}  valid images : {len(valid):>4}  |  "
              f"invalid/non-image files : {len(invalid)}")

    # Report invalid files if any
    any_invalid = any(len(v) > 0 for v in invalid_files.values())
    if any_invalid:
        print()
        print("  ── Invalid / non-image files found (not copied) ──")
        for label, paths in invalid_files.items():
            for p in paths:
                print(f"    [{label}] {os.path.basename(p)}")

    print()

    # ── Step 2: Split ────────────────────────────────────────────────────────
    print("── Step 2: Splitting (stratified, seed=42) ──")

    splits_data = {}   # { class_label: { 'train': [...], 'validation': [...], 'test': [...] } }

    for class_label, files in class_files.items():
        if len(files) == 0:
            print(f"  [WARNING] No valid images found for class '{class_label}'. Skipping.")
            continue

        train, val, test = split_list(
            files,
            TRAIN_RATIO,
            VALIDATION_RATIO,
            RANDOM_SEED
        )
        splits_data[class_label] = {
            "train":      train,
            "validation": val,
            "test":       test,
        }

        print(f"  {class_label:<8}  total {len(files):>4}  →  "
              f"train {len(train):>4}  |  val {len(val):>4}  |  test {len(test):>4}")

    print()

    # ── Step 3: Copy ─────────────────────────────────────────────────────────
    print("── Step 3: Copying images to prepared/ ──")

    summary = {}   # { split: { class: { 'copied': n, 'skipped': n } } }

    for split in SPLITS:
        summary[split] = {}
        for class_label, split_dict in splits_data.items():
            dest_dir = os.path.join(PREPARED_ROOT, split, class_label)
            files    = split_dict[split]
            copied, skipped = copy_files(files, dest_dir)
            summary[split][class_label] = {"copied": copied, "skipped": skipped}

            status = "copied" if skipped == 0 else f"copied {copied}, skipped {skipped} (already exist)"
            print(f"  {split:<12} / {class_label:<8}  → {len(files):>4} images  [{status}]")

    print()

    # ── Step 4: Final report ─────────────────────────────────────────────────
    print("=" * 60)
    print("  FINAL DATASET SUMMARY")
    print("=" * 60)

    total_ripe   = len(class_files.get("ripe",   []))
    total_unripe = len(class_files.get("unripe", []))
    total_all    = total_ripe + total_unripe

    print(f"\n  Total images in original dataset")
    print(f"    Ripe   : {total_ripe}")
    print(f"    Unripe : {total_unripe}")
    print(f"    Total  : {total_all}")
    print()

    col_w = 10
    header = f"  {'Split':<12} {'Ripe':>{col_w}} {'Unripe':>{col_w}} {'Total':>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    grand_total = 0
    for split in SPLITS:
        r = len(splits_data.get("ripe",   {}).get(split, []))
        u = len(splits_data.get("unripe", {}).get(split, []))
        t = r + u
        grand_total += t
        print(f"  {split:<12} {r:>{col_w}} {u:>{col_w}} {t:>{col_w}}")

    print("  " + "-" * (len(header) - 2))
    grand_r = sum(len(splits_data.get("ripe",   {}).get(s, [])) for s in SPLITS)
    grand_u = sum(len(splits_data.get("unripe", {}).get(s, [])) for s in SPLITS)
    print(f"  {'TOTAL':<12} {grand_r:>{col_w}} {grand_u:>{col_w}} {grand_total:>{col_w}}")

    print()
    print("  Prepared dataset saved to:")
    print(f"    {PREPARED_ROOT}")
    print()
    print("  Original dataset is untouched.")
    print("=" * 60)


if __name__ == "__main__":
    main()
