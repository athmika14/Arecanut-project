# -*- coding: utf-8 -*-
"""
prepare_dataset.py  —  ArecaNut AI: Disease Detection
======================================================
Produces a reproducible, source-group-stratified 70 / 15 / 15
train / validation / test split that is free of source-level leakage.

Key decisions (grounded in prior audit reports):
  - ALL three original partitions (train, test, final_testing) are pooled.
    final_testing has ZERO independent sources — every source_id it contains
    already appears in train or test.  Treating it as held-out data would
    be incorrect; merging it into the pool is the only honest choice.
  - Splitting is done at the SOURCE-GROUP level per class, not per image.
    Every augmented copy of a source photograph lands in one partition only.
  - Within each class the 70/15/15 ratio is applied to the SOURCE list,
    then all images belonging to those sources are collected.
  - Random seed = 42 throughout (reproducible).
  - Labels are read from the folder name that contains each file, not
    inferred from the filename prefix, to avoid any ambiguity.
  - Conflicting labels (same source_id mapped to two different classes)
    halt execution with a clear error message.
  - healthy_foot (7 sources) is flagged as too small for reliable
    evaluation; it is included but the warning is printed and recorded.

Outputs (READ-ONLY on originals, no files moved/deleted/renamed):
  ml/disease/prepared_data/
      manifest_train.csv      — all training image rows
      manifest_val.csv        — all validation image rows
      manifest_test.csv       — all test image rows
      manifest_full.csv       — complete dataset with partition column
  ml/disease/results/
      leakage_free_split_report.txt

CSV columns:
  image_path, class_label, source_id, partition

Usage:
    python prepare_dataset.py
Safe to re-run: output files are overwritten (manifests are cheap to
regenerate; original images are never touched).
"""

import os
import re
import sys
import csv
import random
from collections import defaultdict
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ML_DISEASE    = os.path.dirname(SCRIPT_DIR)
DATASET_ROOT  = os.path.join(ML_DISEASE, "dataset")
RESULTS_DIR   = os.path.join(ML_DISEASE, "results")
PREPARED_DIR  = os.path.join(ML_DISEASE, "prepared_data")

TRAIN_DIR = os.path.join(DATASET_ROOT, "Arecanut_dataset",
                         "Arecanut_dataset", "train")
TEST_DIR  = os.path.join(DATASET_ROOT, "Arecanut_dataset",
                         "Arecanut_dataset", "test")
FINAL_DIR = os.path.join(DATASET_ROOT,
                         "final_testing-20230830T042904Z-001",
                         "final_testing")

# Verified from audit reports
EXPECTED_CLASSES = [
    "Healthy_Leaf", "Healthy_Nut", "Healthy_Trunk", "Mahali_Koleroga",
    "Stem_bleeding", "bud borer", "healthy_foot", "stem cracking",
    "yellow leaf disease",
]

# Classes with very few independent sources — flagged but not excluded
LOW_SOURCE_THRESHOLD = 10   # fewer than this triggers a warning

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN - VAL  (remainder, avoids rounding loss)

RANDOM_SEED = 42

UUID_RE  = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
FNAME_RE = re.compile(
    r"^(.+)_original_(.+?)_(" + UUID_RE + r")\.jpg$",
    re.IGNORECASE,
)
VALID_EXTS = {".jpg", ".jpeg", ".png"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_filename(fname):
    """Return (class_prefix, source_id, uuid) or None."""
    m = FNAME_RE.match(fname)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def collect_folder(folder_path, label):
    """
    Walk a single flat folder (one class).
    Returns list of dicts: {image_path, class_label, source_id}.
    Also returns lists: unmatched_files, non_image_files.
    """
    records        = []
    unmatched      = []
    non_image      = []

    if not os.path.isdir(folder_path):
        return records, unmatched, non_image

    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        _, ext = os.path.splitext(fname)
        if ext.lower() not in VALID_EXTS:
            non_image.append(fpath)
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            unmatched.append(fpath)
            continue
        _cls_prefix, source_id, _uuid = parsed
        records.append({
            "image_path":  fpath,
            "class_label": label,       # authoritative: folder name
            "source_id":   source_id,
        })

    return records, unmatched, non_image


def collect_all_sources(lines, p):
    """
    Pool every image from train/, test/, and final_testing/.
    Returns:
      all_records  : list of {image_path, class_label, source_id}
      issues       : list of human-readable issue strings (non-fatal)
      fatal        : list of fatal error strings (abort if non-empty)
    """
    all_records = []
    issues      = []
    fatal       = []

    # ── train/ and test/ — standard class subfolders ──────────────────────
    for split_dir, split_name in [(TRAIN_DIR, "train"), (TEST_DIR, "test")]:
        if not os.path.isdir(split_dir):
            fatal.append(f"Directory not found: {split_dir}")
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            recs, unmatched, non_img = collect_folder(cls_path, cls)
            all_records.extend(recs)
            if unmatched:
                issues.append(
                    f"[{split_name}/{cls}] {len(unmatched)} file(s) did not "
                    f"match the naming pattern — skipped."
                )
            if non_img:
                issues.append(
                    f"[{split_name}/{cls}] {len(non_img)} non-image file(s) "
                    f"skipped."
                )

    # ── final_testing/ — flat files + class subfolders ────────────────────
    if not os.path.isdir(FINAL_DIR):
        fatal.append(f"Directory not found: {FINAL_DIR}")
    else:
        # Flat files: infer class from filename prefix before _original_
        flat_unmatched = []
        flat_non_img   = []
        for fname in sorted(os.listdir(FINAL_DIR)):
            fpath = os.path.join(FINAL_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            _, ext = os.path.splitext(fname)
            if ext.lower() not in VALID_EXTS:
                flat_non_img.append(fpath)
                continue
            parsed = parse_filename(fname)
            if parsed is None:
                flat_unmatched.append(fpath)
                continue
            cls_prefix, source_id, _uuid = parsed
            # Verify the prefix matches a known class
            matched_cls = None
            for known in EXPECTED_CLASSES:
                if known.lower() == cls_prefix.lower():
                    matched_cls = known
                    break
            if matched_cls is None:
                issues.append(
                    f"[final_testing/flat] '{fname}' — inferred class "
                    f"'{cls_prefix}' not in known class list. Skipped."
                )
                continue
            all_records.append({
                "image_path":  fpath,
                "class_label": matched_cls,
                "source_id":   source_id,
            })
        if flat_unmatched:
            issues.append(
                f"[final_testing/flat] {len(flat_unmatched)} file(s) did not "
                f"match naming pattern — skipped."
            )
        if flat_non_img:
            issues.append(
                f"[final_testing/flat] {len(flat_non_img)} non-image file(s) "
                f"skipped."
            )

        # Subfolders
        for entry in sorted(os.listdir(FINAL_DIR)):
            sub_path = os.path.join(FINAL_DIR, entry)
            if not os.path.isdir(sub_path):
                continue
            # Verify subfolder name is a known class
            matched_cls = None
            for known in EXPECTED_CLASSES:
                if known.lower() == entry.lower():
                    matched_cls = known
                    break
            if matched_cls is None:
                issues.append(
                    f"[final_testing] Unexpected subfolder '{entry}' — "
                    f"not a known class. Skipped."
                )
                continue
            recs, unmatched, non_img = collect_folder(sub_path, matched_cls)
            all_records.extend(recs)
            if unmatched:
                issues.append(
                    f"[final_testing/{entry}] {len(unmatched)} file(s) did "
                    f"not match naming pattern — skipped."
                )

    return all_records, issues, fatal


def check_label_conflicts(all_records):
    """
    Verify: for every (class_label, source_id) pair, the class_label is
    consistent across all records that share it.

    Returns list of conflict description strings.
    A conflict = same source_id associated with two different class labels
    (within the same class folder this cannot happen; across different
    class folders it should not happen by dataset design).
    """
    source_to_classes = defaultdict(set)  # source_id -> set of class_labels
    for r in all_records:
        source_to_classes[r["source_id"]].add(r["class_label"])

    conflicts = []
    for src_id, labels in source_to_classes.items():
        if len(labels) > 1:
            conflicts.append(
                f"source_id='{src_id}' mapped to multiple classes: {sorted(labels)}"
            )
    return conflicts


def group_by_class_source(all_records):
    """
    Returns nested dict:
      class_label -> source_id -> [record, ...]
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_records:
        grouped[r["class_label"]][r["source_id"]].append(r)
    return grouped


def stratified_source_split(grouped, train_r, val_r, seed, lines, p):
    """
    For each class independently:
      1. Collect the list of unique source_ids.
      2. Shuffle with fixed seed.
      3. Assign first 70% -> train, next 15% -> val, remainder -> test.
      4. Collect all images for each assigned source into the partition.

    Returns three lists of records: train_recs, val_recs, test_recs.
    Also returns per_class_stats dict for reporting.
    """
    rng = random.Random(seed)

    train_recs  = []
    val_recs    = []
    test_recs   = []
    per_class   = {}
    warnings    = []

    for cls in sorted(grouped.keys()):
        src_dict    = grouped[cls]
        source_ids  = sorted(src_dict.keys())   # sort for determinism
        n_src       = len(source_ids)

        if n_src < 3:
            warnings.append(
                f"  [FATAL-CANDIDATE] Class '{cls}' has only {n_src} source(s). "
                f"Cannot form 3 non-empty partitions."
            )
            # Still include all images in train to avoid data loss
            for src_id in source_ids:
                train_recs.extend(src_dict[src_id])
            per_class[cls] = dict(
                n_src=n_src, tr_src=n_src, va_src=0, te_src=0,
                tr_img=sum(len(v) for v in src_dict.values()),
                va_img=0, te_img=0, warning="TOO FEW SOURCES"
            )
            continue

        if n_src < LOW_SOURCE_THRESHOLD:
            warnings.append(
                f"  [WARNING] Class '{cls}' has only {n_src} source(s) — "
                f"below threshold {LOW_SOURCE_THRESHOLD}. "
                f"Val/test sets will be very small."
            )

        shuffled = source_ids[:]
        rng.shuffle(shuffled)

        n_train = round(n_src * train_r)
        n_val   = round(n_src * val_r)
        n_test  = n_src - n_train - n_val

        # Guard: ensure at least 1 source per val/test
        if n_val < 1:
            n_val   = 1
            n_train = n_src - n_val - n_test
        if n_test < 1:
            n_test  = 1
            n_train = n_src - n_val - n_test

        tr_sources  = shuffled[:n_train]
        va_sources  = shuffled[n_train : n_train + n_val]
        te_sources  = shuffled[n_train + n_val :]

        tr_imgs = [r for s in tr_sources for r in src_dict[s]]
        va_imgs = [r for s in va_sources for r in src_dict[s]]
        te_imgs = [r for s in te_sources for r in src_dict[s]]

        train_recs.extend(tr_imgs)
        val_recs.extend(va_imgs)
        test_recs.extend(te_imgs)

        per_class[cls] = dict(
            n_src=n_src,
            tr_src=len(tr_sources), va_src=len(va_sources), te_src=len(te_sources),
            tr_img=len(tr_imgs),    va_img=len(va_imgs),    te_img=len(te_imgs),
            warning=None,
        )

    return train_recs, val_recs, test_recs, per_class, warnings


def verify_no_source_leakage(train_recs, val_recs, test_recs):
    """
    For each partition pair, check that no (class_label, source_id) pair
    appears in more than one partition.
    Returns list of leakage descriptions (empty = clean).
    """
    def key_set(recs):
        return {(r["class_label"], r["source_id"]) for r in recs}

    tr_keys = key_set(train_recs)
    va_keys = key_set(val_recs)
    te_keys = key_set(test_recs)

    leaks = []
    tr_va = tr_keys & va_keys
    tr_te = tr_keys & te_keys
    va_te = va_keys & te_keys

    for pair in sorted(tr_va):
        leaks.append(f"  TRAIN-VAL leak: class='{pair[0]}' source='{pair[1]}'")
    for pair in sorted(tr_te):
        leaks.append(f"  TRAIN-TEST leak: class='{pair[0]}' source='{pair[1]}'")
    for pair in sorted(va_te):
        leaks.append(f"  VAL-TEST leak: class='{pair[0]}' source='{pair[1]}'")

    return leaks


def write_csv(records, partition_label, filepath):
    """Write records to a CSV file with a partition column."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "class_label", "source_id", "partition"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "image_path":  r["image_path"],
                "class_label": r["class_label"],
                "source_id":   r["source_id"],
                "partition":   partition_label,
            })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR,  exist_ok=True)
    os.makedirs(PREPARED_DIR, exist_ok=True)

    lines = []

    def p(text=""):
        print(text)
        lines.append(text)

    # ── Header ────────────────────────────────────────────────────────────────
    p("=" * 72)
    p("  ArecaNut AI — Disease Detection: Leakage-Free Dataset Split")
    p(f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    p("=" * 72)
    p()
    p("  Strategy  : Group-stratified split at source-ID level per class")
    p("  Ratio     : 70% train / 15% validation / 15% test")
    p("  Seed      : 42 (fixed, reproducible)")
    p("  Pool      : train + test + final_testing (all merged — final_testing")
    p("              has zero independent sources per source_leakage_analysis)")
    p("  Originals : READ-ONLY — no files moved, renamed, or deleted")

    # ── Step 0: Directory checks ───────────────────────────────────────────────
    p()
    p("─" * 72)
    p("  STEP 0 — Directory Verification")
    p("─" * 72)
    abort = False
    for label, path in [("TRAIN",         TRAIN_DIR),
                        ("TEST",          TEST_DIR),
                        ("FINAL_TESTING", FINAL_DIR)]:
        ok = os.path.isdir(path)
        p(f"  [{'OK' if ok else 'MISSING'}]  {label}: {path}")
        if not ok:
            abort = True
    if abort:
        p()
        p("  ABORT: Required directories are missing. Cannot continue.")
        _save_report(lines, RESULTS_DIR)
        sys.exit(1)

    # ── Step 1: Collect all records ───────────────────────────────────────────
    p()
    p("─" * 72)
    p("  STEP 1 — Collecting All Image Records")
    p("─" * 72)

    all_records, issues, fatal = collect_all_sources(lines, p)

    if fatal:
        p()
        p("  FATAL ERRORS:")
        for f in fatal:
            p(f"    {f}")
        p("  Cannot continue.")
        _save_report(lines, RESULTS_DIR)
        sys.exit(1)

    p()
    p(f"  Total image records collected : {len(all_records)}")

    if issues:
        p()
        p("  Issues encountered during collection (non-fatal):")
        for iss in issues:
            p(f"    {iss}")
    else:
        p("  No collection issues.")

    # ── Step 2: Verify class labels ───────────────────────────────────────────
    p()
    p("─" * 72)
    p("  STEP 2 — Class Label Verification")
    p("─" * 72)

    found_classes = sorted({r["class_label"] for r in all_records})
    p()
    p(f"  Classes found in pool : {len(found_classes)}")
    for cls in found_classes:
        n = sum(1 for r in all_records if r["class_label"] == cls)
        p(f"    {cls:<30}  {n:>5} images")

    missing = [c for c in EXPECTED_CLASSES if c not in found_classes]
    extra   = [c for c in found_classes if c not in EXPECTED_CLASSES]
    if missing:
        p()
        p(f"  [WARNING] Expected classes not found: {missing}")
    if extra:
        p()
        p(f"  [WARNING] Unexpected classes found: {extra}")
    if not missing and not extra:
        p()
        p("  [OK] All 9 expected classes present, no unexpected classes.")

    # ── Step 3: Check for conflicting labels ──────────────────────────────────
    p()
    p("─" * 72)
    p("  STEP 3 — Source-ID Label Conflict Check")
    p("─" * 72)

    conflicts = check_label_conflicts(all_records)
    p()
    if conflicts:
        p(f"  [FATAL] {len(conflicts)} source_id(s) are mapped to multiple "
          f"class labels:")
        for c in conflicts:
            p(f"    {c}")
        p()
        p("  Cannot build a reliable split. Aborting.")
        _save_report(lines, RESULTS_DIR)
        sys.exit(1)
    else:
        p("  [OK] No source_id is associated with more than one class label.")

    # ── Step 4: Group by class and source ─────────────────────────────────────
    p()
    p("─" * 72)
    p("  STEP 4 — Source Groups Per Class")
    p("─" * 72)
    p()

    grouped = group_by_class_source(all_records)

    col = max(len(c) for c in found_classes) + 2
    hdr = (f"  {'Class':<{col}}  {'Sources':>8}  {'Images':>8}  "
           f"{'Avg/Src':>8}  {'Min':>5}  {'Max':>5}")
    p(hdr)
    p("  " + "-" * (len(hdr) - 2))

    grand_src = 0; grand_img = 0
    for cls in sorted(grouped.keys()):
        src_dict = grouped[cls]
        n_src    = len(src_dict)
        n_img    = sum(len(v) for v in src_dict.values())
        avg      = n_img / n_src if n_src else 0
        mn       = min(len(v) for v in src_dict.values()) if src_dict else 0
        mx       = max(len(v) for v in src_dict.values()) if src_dict else 0
        grand_src += n_src; grand_img += n_img
        flag = "  <-- LOW SOURCE COUNT" if n_src < LOW_SOURCE_THRESHOLD else ""
        p(f"  {cls:<{col}}  {n_src:>8}  {n_img:>8}  "
          f"{avg:>8.1f}  {mn:>5}  {mx:>5}{flag}")

    p("  " + "-" * (len(hdr) - 2))
    p(f"  {'TOTAL':<{col}}  {grand_src:>8}  {grand_img:>8}")

    # ── Step 5: Perform split ─────────────────────────────────────────────────
    p()
    p("─" * 72)
    p("  STEP 5 — Source-Group Stratified Split (seed=42)")
    p("─" * 72)

    train_recs, val_recs, test_recs, per_class, split_warnings = \
        stratified_source_split(grouped, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
                                lines, p)

    if split_warnings:
        p()
        p("  Split warnings:")
        for w in split_warnings:
            p(w)

    # ── Step 6: Per-class split distribution ──────────────────────────────────
    p()
    p("─" * 72)
    p("  STEP 6 — Per-Class Split Distribution")
    p("─" * 72)
    p()
    p("  Source counts:")
    hdr2 = (f"  {'Class':<{col}}  "
            f"{'Total-Src':>10}  {'Tr-Src':>7}  {'Va-Src':>7}  {'Te-Src':>7}  "
            f"{'Tr%':>5}")
    p(hdr2)
    p("  " + "-" * (len(hdr2) - 2))
    g_tsrc = g_trsrc = g_vasrc = g_tesrc = 0
    for cls in sorted(per_class.keys()):
        s = per_class[cls]
        tr_pct = s['tr_src'] / s['n_src'] * 100 if s['n_src'] else 0
        warn_tag = f"  [{s['warning']}]" if s['warning'] else ""
        p(f"  {cls:<{col}}  "
          f"{s['n_src']:>10}  {s['tr_src']:>7}  {s['va_src']:>7}  "
          f"{s['te_src']:>7}  {tr_pct:>4.0f}%{warn_tag}")
        g_tsrc += s['n_src']; g_trsrc += s['tr_src']
        g_vasrc += s['va_src']; g_tesrc += s['te_src']
    p("  " + "-" * (len(hdr2) - 2))
    p(f"  {'TOTAL':<{col}}  {g_tsrc:>10}  {g_trsrc:>7}  {g_vasrc:>7}  {g_tesrc:>7}")

    p()
    p("  Image counts:")
    hdr3 = (f"  {'Class':<{col}}  "
            f"{'Total-Img':>10}  {'Tr-Img':>7}  {'Va-Img':>7}  {'Te-Img':>7}  "
            f"{'Tr%':>5}")
    p(hdr3)
    p("  " + "-" * (len(hdr3) - 2))
    g_timg = g_trimg = g_vaimg = g_teimg = 0
    for cls in sorted(per_class.keys()):
        s = per_class[cls]
        total_img = s['tr_img'] + s['va_img'] + s['te_img']
        tr_pct    = s['tr_img'] / total_img * 100 if total_img else 0
        p(f"  {cls:<{col}}  "
          f"{total_img:>10}  {s['tr_img']:>7}  {s['va_img']:>7}  "
          f"{s['te_img']:>7}  {tr_pct:>4.0f}%")
        g_timg += total_img; g_trimg += s['tr_img']
        g_vaimg += s['va_img']; g_teimg += s['te_img']
    p("  " + "-" * (len(hdr3) - 2))
    p(f"  {'TOTAL':<{col}}  {g_timg:>10}  {g_trimg:>7}  {g_vaimg:>7}  {g_teimg:>7}")
    p()
    p(f"  Achieved ratios (images):")
    p(f"    Train      : {g_trimg:>5}  ({g_trimg/g_timg*100:.1f}%)")
    p(f"    Validation : {g_vaimg:>5}  ({g_vaimg/g_timg*100:.1f}%)")
    p(f"    Test       : {g_teimg:>5}  ({g_teimg/g_timg*100:.1f}%)")

    # ── Step 7: Leakage verification ──────────────────────────────────────────
    p()
    p("─" * 72)
    p("  STEP 7 — Source-Group Leakage Verification")
    p("─" * 72)
    p()
    p("  Checking that no (class, source_id) key appears in more than")
    p("  one partition...")
    p()

    leaks = verify_no_source_leakage(train_recs, val_recs, test_recs)

    if leaks:
        p(f"  [FAIL] {len(leaks)} source-group leak(s) detected:")
        for lk in leaks:
            p(lk)
        p()
        p("  This is a bug in the split logic. Aborting without saving CSVs.")
        _save_report(lines, RESULTS_DIR)
        sys.exit(1)
    else:
        p("  [PASS] Zero source-group leaks across all three partitions.")
        p("         Every source_id is assigned to exactly one of:")
        p("         train / validation / test.")

    # ── Step 8: Write CSV manifests ───────────────────────────────────────────
    p()
    p("─" * 72)
    p("  STEP 8 — Writing CSV Manifests")
    p("─" * 72)
    p()

    csv_train = os.path.join(PREPARED_DIR, "manifest_train.csv")
    csv_val   = os.path.join(PREPARED_DIR, "manifest_val.csv")
    csv_test  = os.path.join(PREPARED_DIR, "manifest_test.csv")
    csv_full  = os.path.join(PREPARED_DIR, "manifest_full.csv")

    write_csv(train_recs, "train",      csv_train)
    write_csv(val_recs,   "validation", csv_val)
    write_csv(test_recs,  "test",       csv_test)

    # Full manifest: all partitions in one file
    all_labelled = (
        [(r, "train")      for r in train_recs] +
        [(r, "validation") for r in val_recs]   +
        [(r, "test")       for r in test_recs]
    )
    os.makedirs(PREPARED_DIR, exist_ok=True)
    with open(csv_full, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "class_label", "source_id", "partition"]
        )
        writer.writeheader()
        for r, part in all_labelled:
            writer.writerow({
                "image_path":  r["image_path"],
                "class_label": r["class_label"],
                "source_id":   r["source_id"],
                "partition":   part,
            })

    for fpath, label, count in [
        (csv_train, "manifest_train.csv", len(train_recs)),
        (csv_val,   "manifest_val.csv",   len(val_recs)),
        (csv_test,  "manifest_test.csv",  len(test_recs)),
        (csv_full,  "manifest_full.csv",  len(all_labelled)),
    ]:
        size_kb = os.path.getsize(fpath) / 1024
        p(f"  {label:<25}  {count:>6} rows  ({size_kb:.1f} KB)")
        p(f"    {fpath}")

    # ── Step 9: Class imbalance summary ───────────────────────────────────────
    p()
    p("─" * 72)
    p("  STEP 9 — Class Imbalance Summary (Training Set)")
    p("─" * 72)
    p()

    train_class_counts = defaultdict(int)
    for r in train_recs:
        train_class_counts[r["class_label"]] += 1

    total_train_imgs = sum(train_class_counts.values())
    sorted_by_count  = sorted(train_class_counts.items(),
                               key=lambda x: x[1], reverse=True)
    max_count = sorted_by_count[0][1]
    min_count = sorted_by_count[-1][1]

    for cls, cnt in sorted_by_count:
        bar = "#" * max(1, round(cnt / max_count * 40))
        pct = cnt / total_train_imgs * 100
        p(f"  {cls:<30}  {cnt:>5} imgs  {pct:>5.1f}%  {bar}")

    p()
    p(f"  Imbalance ratio (max:min) : {max_count}:{min_count}  "
      f"(~{round(max_count/min_count)}:1)")
    p()
    p("  Recommended mitigation during training (no file changes needed):")
    p("    keras:  class_weight=  computed with sklearn.utils.class_weight")
    p("    keras:  or use tf.data weighted sampling")
    p("    loss:   alternatively use focal loss")

    # ── Summary ───────────────────────────────────────────────────────────────
    p()
    p("=" * 72)
    p("  SUMMARY")
    p("=" * 72)
    p()
    p("  [VERIFIED]")
    p(f"  Total images pooled from all 3 original partitions : {len(all_records)}")
    p(f"  Unique source groups across all classes            : {grand_src}")
    p(f"  Label conflicts detected                           : 0")
    p(f"  Source-group leaks in new split                    : 0")
    p()
    p("  New split image counts:")
    p(f"    Train      : {g_trimg:>6}  ({g_trimg/g_timg*100:.1f}%)")
    p(f"    Validation : {g_vaimg:>6}  ({g_vaimg/g_timg*100:.1f}%)")
    p(f"    Test       : {g_teimg:>6}  ({g_teimg/g_timg*100:.1f}%)")
    p(f"    Total      : {g_timg:>6}")
    p()
    p("  New split source counts:")
    p(f"    Train      : {g_trsrc:>6}")
    p(f"    Validation : {g_vasrc:>6}")
    p(f"    Test       : {g_tesrc:>6}")
    p(f"    Total      : {g_tsrc:>6}")
    p()

    # Low-source-count classes
    low_classes = [cls for cls, s in per_class.items()
                   if s["n_src"] < LOW_SOURCE_THRESHOLD]
    if low_classes:
        p("  [WARNING] Classes with too few independent sources for")
        p(f"  reliable evaluation (< {LOW_SOURCE_THRESHOLD} source groups):")
        for cls in low_classes:
            s = per_class[cls]
            p(f"    {cls:<30}  {s['n_src']} sources  "
              f"val={s['va_src']}  test={s['te_src']}")
        p()
        p("  These classes are included in the split but their individual")
        p("  per-class precision/recall in val/test may not be meaningful.")

    p()
    p("  Output files:")
    p(f"    {csv_train}")
    p(f"    {csv_val}")
    p(f"    {csv_test}")
    p(f"    {csv_full}")
    report_path = os.path.join(RESULTS_DIR, "leakage_free_split_report.txt")
    p(f"    {report_path}")
    p()
    p("=" * 72)

    _save_report(lines, RESULTS_DIR)


def _save_report(lines, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "leakage_free_split_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Report written: {path}")


if __name__ == "__main__":
    main()
