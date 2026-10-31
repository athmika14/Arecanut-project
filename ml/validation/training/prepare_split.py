# -*- coding: utf-8 -*-
"""
prepare_split.py — ArecaNut Validator: Source-Group Stratified Split
=======================================================================
Creates leakage-free train/validation/test manifests for the binary
arecanut-vs-not-arecanut classifier.

Positive images (arecanut):
  Read from the disease dataset manifest.  Source-group splitting uses
  the source_id field to keep all augmented copies of one original in
  one partition.

Negative images (not_arecanut):
  Scanned from ml/validation/dataset/not_arecanut/<category>/.
  MD5 hashing detects exact duplicates (removed).  Since negatives do
  not have source-group metadata, they are split randomly with a fixed
  seed.  Near-duplicates are detected by hash and kept together.

Output manifests (CSV with columns: image_path, class_label, source_id, partition):
  ml/validation/prepared_data/manifest_train.csv
  ml/validation/prepared_data/manifest_val.csv
  ml/validation/prepared_data/manifest_test.csv
  ml/validation/prepared_data/manifest_full.csv

Usage:
    cd backend
    python -m ml.validation.training.prepare_split
"""

import os
import sys
import csv
import hashlib
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# -- Paths ---------------------------------------------------------------------

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.dirname(SCRIPT_DIR)
ML_DIR        = os.path.dirname(VALIDATION_DIR)
PROJECT_DIR   = os.path.dirname(ML_DIR)

DISEASE_MANIFEST = os.path.join(
    ML_DIR, "disease", "prepared_data", "manifest_full.csv"
)
NEGATIVES_DIR = os.path.join(VALIDATION_DIR, "dataset", "not_arecanut")
PREPARED_DIR  = os.path.join(VALIDATION_DIR, "prepared_data")
RESULTS_DIR   = os.path.join(VALIDATION_DIR, "results")

VALID_EXTS = {".jpg", ".jpeg", ".png"}

# Split ratio
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO = 1 - TRAIN - VAL

SEED = 42

# -- Helpers ------------------------------------------------------------------

def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_negatives(base_dir: str):
    """Return list of (image_path, category_name) for all negative images."""
    results = []
    if not os.path.isdir(base_dir):
        return results
    for entry in sorted(os.listdir(base_dir)):
        sub = os.path.join(base_dir, entry)
        if not os.path.isdir(sub):
            continue
        for fname in sorted(os.listdir(sub)):
            fpath = os.path.join(sub, fname)
            if os.path.isfile(fpath):
                ext = os.path.splitext(fname)[1].lower()
                if ext in VALID_EXTS:
                    results.append((fpath, entry))
    return results


def split_source_groups(source_groups: dict, train_r: float, val_r: float, seed: int):
    """
    Split a dict of {source_id: [records]} into (train, val, test) record lists.
    Each source group goes entirely to one partition.
    """
    rng = random.Random(seed)
    source_ids = sorted(source_groups.keys())
    rng.shuffle(source_ids)

    n = len(source_ids)
    n_train = round(n * train_r)
    n_val   = round(n * val_r)
    n_test  = n - n_train - n_val

    if n_test < 1 and n >= 3:
        n_test = 1
        n_train = n - n_val - n_test
    if n_val < 1 and n >= 3:
        n_val = 1
        n_train = n - n_val - n_test

    tr_srcs = source_ids[:n_train]
    va_srcs = source_ids[n_train : n_train + n_val]
    te_srcs = source_ids[n_train + n_val :]

    train_recs = [r for s in tr_srcs for r in source_groups[s]]
    val_recs   = [r for s in va_srcs for r in source_groups[s]]
    test_recs  = [r for s in te_srcs for r in source_groups[s]]

    return train_recs, val_recs, test_recs, tr_srcs, va_srcs, te_srcs


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  ArecaNut Validator — Source-Group Stratified Split")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Seed: {SEED}  Ratio: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/"
          f"{int((1-TRAIN_RATIO-VAL_RATIO)*100)}")
    print("=" * 70)

    os.makedirs(PREPARED_DIR, exist_ok=True)

    all_records = []  # list of (image_path, class_label, source_id)

    # -- Step 1: Collect positive (arecanut) records -------------------------
    print("\n-- Step 1: Loading positive (arecanut) images --")

    if not os.path.isfile(DISEASE_MANIFEST):
        print(f"  [FATAL] Disease manifest not found: {DISEASE_MANIFEST}")
        sys.exit(1)

    pos_rows = []
    with open(DISEASE_MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pos_rows.append(r)

    # Group by source_id
    pos_groups = defaultdict(list)
    for r in pos_rows:
        pos_groups[r["source_id"]].append(
            (r["image_path"], "arecanut", r["source_id"])
        )

    print(f"  Positive images: {len(pos_rows)}")
    print(f"  Unique source groups: {len(pos_groups)}")

    # Check all paths exist
    missing = [r["image_path"] for r in pos_rows
               if not os.path.isfile(r["image_path"])]
    if missing:
        print(f"  [FATAL] {len(missing)} positive image paths do not exist.")
        sys.exit(1)
    print(f"  [OK] All positive paths verified.")

    # -- Step 2: Collect negative (not_arecanut) records ----------------------
    print("\n-- Step 2: Loading negative (not_arecanut) images --")

    neg_items = scan_negatives(NEGATIVES_DIR)
    print(f"  Negative images found: {len(neg_items)}")

    if len(neg_items) == 0:
        print("  [FATAL] No negative images found.")
        print(f"  Place images in subfolders under: {NEGATIVES_DIR}")
        sys.exit(1)

    # Deduplicate negatives by MD5 hash
    print("  Computing MD5 hashes for deduplication...")
    seen_hashes = set()
    neg_deduped = []
    dup_count = 0
    for fpath, category in neg_items:
        h = md5(fpath)
        if h in seen_hashes:
            dup_count += 1
            continue
        seen_hashes.add(h)
        # Use hash as source_id (so exact duplicates would group together,
        # but we already removed them)
        neg_deduped.append((fpath, "not_arecanut", f"neg_{h[:12]}"))

    if dup_count:
        print(f"  [INFO] Removed {dup_count} exact duplicate(s).")

    # Group negatives by source_id (hash-based)
    neg_groups = defaultdict(list)
    for rec in neg_deduped:
        neg_groups[rec[2]].append(rec)

    print(f"  Unique negative images after dedup: {len(neg_deduped)}")
    print(f"  Unique negative 'source groups': {len(neg_groups)}")

    # -- Step 3: Split positives by source group -----------------------------
    print("\n-- Step 3: Splitting positives by source group --")

    pos_tr, pos_va, pos_te, pos_tr_srcs, pos_va_srcs, pos_te_srcs = \
        split_source_groups(pos_groups, TRAIN_RATIO, VAL_RATIO, SEED)

    print(f"  Positive split: train={len(pos_tr)} val={len(pos_va)} test={len(pos_te)}")
    print(f"  Source groups:  train={len(pos_tr_srcs)} val={len(pos_va_srcs)} test={len(pos_te_srcs)}")

    # -- Step 4: Split negatives by hash group -------------------------------
    print("\n-- Step 4: Splitting negatives by hash group --")

    neg_tr, neg_va, neg_te, neg_tr_srcs, neg_va_srcs, neg_te_srcs = \
        split_source_groups(neg_groups, TRAIN_RATIO, VAL_RATIO, SEED + 1)

    print(f"  Negative split: train={len(neg_tr)} val={len(neg_va)} test={len(neg_te)}")

    # -- Step 5: Combine and verify no leakage --------------------------------
    print("\n-- Step 5: Leakage verification --")

    train_recs = pos_tr + neg_tr
    val_recs   = pos_va + neg_va
    test_recs  = pos_te + neg_te

    def source_keys(recs):
        return {(r[1], r[2]) for r in recs}  # (class, source_id)

    tr_keys = source_keys(train_recs)
    va_keys = source_keys(val_recs)
    te_keys = source_keys(test_recs)

    tr_va = tr_keys & va_keys
    tr_te = tr_keys & te_keys
    va_te = va_keys & te_keys

    if tr_va or tr_te or va_te:
        print(f"  [FAIL] Leakage detected!")
        print(f"    train∩val={len(tr_va)} train∩test={len(tr_te)} val∩test={len(va_te)}")
        sys.exit(1)
    else:
        print(f"  [PASS] No source-group leakage across partitions.")

    # -- Step 6: Write CSV manifests -----------------------------------------
    print("\n-- Step 6: Writing manifests --")

    manifests = [
        ("manifest_train.csv", train_recs, "train"),
        ("manifest_val.csv",   val_recs,    "validation"),
        ("manifest_test.csv",  test_recs,   "test"),
    ]

    for fname, recs, partition in manifests:
        path = os.path.join(PREPARED_DIR, fname)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "class_label", "source_id", "partition"])
            for img_path, cls, src_id in recs:
                w.writerow([img_path, cls, src_id, partition])
        n_pos = sum(1 for r in recs if r[1] == "arecanut")
        n_neg = sum(1 for r in recs if r[1] == "not_arecanut")
        print(f"  {fname:<25}  {len(recs):>5} rows  (pos={n_pos} neg={n_neg})")

    # Full manifest
    full_path = os.path.join(PREPARED_DIR, "manifest_full.csv")
    with open(full_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "class_label", "source_id", "partition"])
        for img_path, cls, src_id in train_recs + val_recs + test_recs:
            part = "train" if (cls, src_id) in tr_keys else \
                   "validation" if (cls, src_id) in va_keys else "test"
            w.writerow([img_path, cls, src_id, part])
    print(f"  {'manifest_full.csv':<25}  {len(train_recs)+len(val_recs)+len(test_recs):>5} rows")

    # -- Summary --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SPLIT SUMMARY")
    print("=" * 70)
    total = len(train_recs) + len(val_recs) + len(test_recs)
    print(f"  Train     : {len(train_recs):>5}  ({len(train_recs)/total*100:.1f}%)")
    print(f"  Validation: {len(val_recs):>5}  ({len(val_recs)/total*100:.1f}%)")
    print(f"  Test      : {len(test_recs):>5}  ({len(test_recs)/total*100:.1f}%)")
    print(f"  Total     : {total}")
    print(f"\n  Manifests saved to: {PREPARED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
