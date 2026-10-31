# -*- coding: utf-8 -*-
"""
audit_dataset.py — ArecaNut Validator: Dataset Audit
=====================================================
Audits the positive (arecanut) and negative (not_arecanut) image
collections and reports counts, duplicates, and class balance.

READ-ONLY: does not modify any files.

Positive images are read from the existing disease dataset manifest
(ml/disease/prepared_data/manifest_full.csv).  Negative images are
scanned from ml/validation/dataset/not_arecanut/<category>/.

Usage:
    cd backend
    python -m ml.validation.training.audit_dataset

    or:

    python ml/validation/training/audit_dataset.py
"""

import os
import sys
import csv
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# -- Paths ---------------------------------------------------------------------

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.dirname(SCRIPT_DIR)         # ml/validation/
ML_DIR        = os.path.dirname(VALIDATION_DIR)      # ml/
PROJECT_DIR   = os.path.dirname(ML_DIR)             # ArecaNut_AI/

DISEASE_MANIFEST = os.path.join(
    ML_DIR, "disease", "prepared_data", "manifest_full.csv"
)

NEGATIVES_DIR = os.path.join(VALIDATION_DIR, "dataset", "not_arecanut")
RESULTS_DIR   = os.path.join(VALIDATION_DIR, "results")

VALID_EXTS = {".jpg", ".jpeg", ".png"}

# Negative image categories expected under not_arecanut/
EXPECTED_CATEGORIES = [
    "people", "animals", "other_plants", "buildings",
    "vehicles", "food", "objects", "scenery",
]

# Minimum recommended images per category
MIN_PER_CATEGORY = {
    "people": 50, "animals": 40, "other_plants": 50,
    "buildings": 30, "vehicles": 30, "food": 30,
    "objects": 40, "scenery": 30,
}
MIN_TOTAL = 300


# -- Helpers ------------------------------------------------------------------

def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_negatives(base_dir: str) -> dict:
    """
    Scan not_arecanut/ subfolders.
    Returns dict: category -> list of image paths.
    """
    categories = {}
    if not os.path.isdir(base_dir):
        return categories
    for entry in sorted(os.listdir(base_dir)):
        sub = os.path.join(base_dir, entry)
        if not os.path.isdir(sub):
            continue
        imgs = []
        for fname in sorted(os.listdir(sub)):
            fpath = os.path.join(sub, fname)
            if os.path.isfile(fpath):
                ext = os.path.splitext(fname)[1].lower()
                if ext in VALID_EXTS:
                    imgs.append(fpath)
        if imgs:
            categories[entry] = imgs
    return categories


def main():
    print("=" * 70)
    print("  ArecaNut Validator — Dataset Audit")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # -- Section 1: Positive images (arecanut) --------------------------------
    print("\n" + "-" * 70)
    print("  SECTION 1 — Positive Images (arecanut)")
    print("-" * 70)

    if not os.path.isfile(DISEASE_MANIFEST):
        print(f"  [ERROR] Disease manifest not found: {DISEASE_MANIFEST}")
        print("  Run the disease dataset preparation first.")
        return

    pos_rows = []
    with open(DISEASE_MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pos_rows.append(r)

    pos_sources = defaultdict(set)   # class -> set of source_ids
    for r in pos_rows:
        pos_sources[r["class_label"]].add(r["source_id"])

    total_pos = len(pos_rows)
    total_sources = len(set(r["source_id"] for r in pos_rows))
    print(f"\n  Total positive images    : {total_pos}")
    print(f"  Unique source groups      : {total_sources}")
    print(f"\n  Per-class breakdown:")
    for cls in sorted(pos_sources.keys()):
        n_img = sum(1 for r in pos_rows if r["class_label"] == cls)
        n_src = len(pos_sources[cls])
        print(f"    {cls:<25}  images={n_img:>5}  sources={n_src:>4}")

    # Check all paths exist
    missing = [r["image_path"] for r in pos_rows
               if not os.path.isfile(r["image_path"])]
    if missing:
        print(f"\n  [WARNING] {len(missing)} positive image paths do not exist on disk.")
    else:
        print(f"\n  [OK] All {total_pos} positive image paths exist on disk.")

    # -- Section 2: Negative images (not_arecanut) ----------------------------
    print("\n" + "-" * 70)
    print("  SECTION 2 — Negative Images (not_arecanut)")
    print("-" * 70)

    neg_cats = scan_negatives(NEGATIVES_DIR)
    total_neg = sum(len(v) for v in neg_cats.values())

    print(f"\n  Negatives directory: {NEGATIVES_DIR}")
    print(f"  Total negative images : {total_neg}")
    print(f"  Categories found      : {len(neg_cats)}")

    if neg_cats:
        print(f"\n  Per-category breakdown:")
        for cat in sorted(neg_cats.keys()):
            n = len(neg_cats[cat])
            min_req = MIN_PER_CATEGORY.get(cat, "?")
            flag = ""
            if isinstance(min_req, int) and n < min_req:
                flag = f"  <-- below minimum ({min_req} recommended)"
            print(f"    {cat:<20}  images={n:>4}{flag}")
    else:
        print("\n  [WARNING] No negative images found.")
        print("  Create subfolders under ml/validation/dataset/not_arecanut/")
        print("  and place images in each category folder.")

    # Check for missing categories
    missing_cats = [c for c in EXPECTED_CATEGORIES if c not in neg_cats]
    if missing_cats:
        print(f"\n  [INFO] Missing categories (create these folders and add images):")
        for c in missing_cats:
            min_req = MIN_PER_CATEGORY.get(c, 30)
            print(f"    {c}/  (minimum {min_req} images recommended)")

    # -- Section 3: Class balance ----------------------------------------------
    print("\n" + "-" * 70)
    print("  SECTION 3 — Class Balance")
    print("-" * 70)

    print(f"\n  Positive (arecanut)     : {total_pos:>6}")
    print(f"  Negative (not_arecanut) : {total_neg:>6}")

    if total_neg == 0:
        ratio = "infinity"
        print(f"\n  [CRITICAL] No negative images. Training cannot proceed.")
        print(f"  Collect at least {MIN_TOTAL} negative images across categories.")
    else:
        ratio = total_pos / total_neg
        print(f"  Ratio (pos:neg)         : {ratio:.1f}:1")
        if ratio > 10:
            print(f"\n  [WARNING] Severe class imbalance (ratio {ratio:.0f}:1).")
            print(f"  Consider collecting more negatives or using class_weight.")
        elif ratio > 3:
            print(f"\n  [INFO] Moderate class imbalance. class_weight will handle it.")
        else:
            print(f"\n  [OK] Reasonable balance.")

    # -- Section 4: Duplicate detection in negatives ---------------------------
    print("\n" + "-" * 70)
    print("  SECTION 4 — Duplicate Detection (negatives)")
    print("-" * 70)

    if total_neg > 0:
        print(f"\n  Computing MD5 hashes for {total_neg} negative images...")
        hashes = defaultdict(list)   # hash -> [paths]
        all_neg_paths = [p for paths in neg_cats.values() for p in paths]
        for p in all_neg_paths:
            h = md5(p)
            hashes[h].append(p)

        exact_dups = {h: paths for h, paths in hashes.items() if len(paths) > 1}
        if exact_dups:
            print(f"\n  [WARNING] {len(exact_dups)} exact duplicate group(s) found:")
            for h, paths in list(exact_dups.items())[:5]:
                print(f"    Hash {h[:12]}...")
                for p in paths:
                    print(f"      {os.path.basename(p)}")
            if len(exact_dups) > 5:
                print(f"    ... and {len(exact_dups) - 5} more duplicate group(s).")
            print(f"\n  Remove duplicates before proceeding to the split step.")
        else:
            print(f"\n  [OK] No exact duplicates found among {total_neg} negative images.")
    else:
        print(f"\n  [SKIP] No negative images to check for duplicates.")

    # -- Summary --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  AUDIT SUMMARY")
    print("=" * 70)
    print(f"\n  Positive (arecanut)     : {total_pos:>6} images  ({total_sources} source groups)")
    print(f"  Negative (not_arecanut) : {total_neg:>6} images  ({len(neg_cats)} categories)")

    ready = True
    if total_neg < MIN_TOTAL:
        print(f"\n  [NOT READY] Only {total_neg} negative images. Minimum: {MIN_TOTAL}.")
        ready = False
    if missing_cats:
        print(f"  [NOT READY] {len(missing_cats)} category folder(s) missing or empty.")
        ready = False
    if exact_dups if total_neg > 0 else False:
        print(f"  [NOT READY] {len(exact_dups)} duplicate group(s) must be removed.")
        ready = False

    if ready:
        print(f"\n  [READY] Dataset is sufficient to proceed to prepare_split.py.")
    else:
        print(f"\n  [ACTION NEEDED] See the instructions above before proceeding.")

    print("=" * 70)

    # Save report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "validator_dataset_audit.txt")
    # Re-run with output capture would be better, but for now just print
    print(f"\n  Report printed to terminal. Save manually if needed: {report_path}")


if __name__ == "__main__":
    main()
