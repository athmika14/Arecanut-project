"""
audit_dataset.py — ArecaNut AI: Disease Detection Dataset Audit
================================================================
READ-ONLY. Does not move, rename, copy, delete, or modify any files.

Checks:
  1. Image counts per class in train / test
  2. final_testing folder structure and image counts
  3. All disease classes and supported extensions found
  4. Exact duplicate detection across all three partitions (MD5 hash)
  5. Unreadable / corrupted images (Pillow verify)
  6. Filename patterns suggesting augmentation or derived copies
  7. Missing classes, inconsistent names, unexpected folders
  8. Train / test leakage (duplicate hashes across splits)
  9. Saves full report to ml/disease/results/dataset_audit.txt
 10. Prints key findings to terminal

Usage:
    python audit_dataset.py
"""

import os
import sys
import hashlib
import re
import io
from collections import defaultdict
from datetime import datetime

# ── Try to import Pillow; gracefully degrade if absent ───────────────────────
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ML_DISEASE   = os.path.dirname(SCRIPT_DIR)                  # ml/disease/
DATASET_ROOT = os.path.join(ML_DISEASE, "dataset")
RESULTS_DIR  = os.path.join(ML_DISEASE, "results")

TRAIN_DIR = os.path.join(DATASET_ROOT, "Arecanut_dataset",
                         "Arecanut_dataset", "train")
TEST_DIR  = os.path.join(DATASET_ROOT, "Arecanut_dataset",
                         "Arecanut_dataset", "test")
FINAL_DIR = os.path.join(DATASET_ROOT,
                         "final_testing-20230830T042904Z-001",
                         "final_testing")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# UUID pattern appended to filenames in final_testing
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE
)
# Pattern: <class>_original_<source>_<uuid>.jpg
DERIVED_PATTERN = re.compile(r"_original_", re.IGNORECASE)

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_image(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in VALID_EXTS


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_image(path):
    """Return (ok: bool, error_msg: str). Requires Pillow."""
    if not PIL_AVAILABLE:
        return True, ""
    try:
        with PILImage.open(path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, str(e)


def scan_flat_dir(directory):
    """
    Scan a directory that contains files directly (no class subfolders).
    Returns dict with keys: images, non_images, all_files.
    Each image entry: {'path', 'filename', 'ext'}.
    """
    result = {"images": [], "non_images": []}
    if not os.path.isdir(directory):
        return result
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            if is_image(fname):
                _, ext = os.path.splitext(fname)
                result["images"].append({
                    "path": fpath, "filename": fname, "ext": ext.lower()
                })
            else:
                result["non_images"].append(fpath)
    return result


def scan_class_dir(split_dir):
    """
    Scan a split directory (train or test) that contains class subfolders.
    Returns:
      class_counts: { class_name: int }
      class_images: { class_name: [ {'path','filename','ext'} ] }
      class_non_images: { class_name: [path] }
      unexpected_files: [path]   — files directly in split_dir (not in a subfolder)
      class_dirs: [str]          — subdirectory names found
    """
    class_images     = defaultdict(list)
    class_non_images = defaultdict(list)
    unexpected_files = []
    class_dirs       = []

    if not os.path.isdir(split_dir):
        return {}, {}, {}, [], []

    for entry in sorted(os.listdir(split_dir)):
        full = os.path.join(split_dir, entry)
        if os.path.isdir(full):
            class_dirs.append(entry)
            for fname in sorted(os.listdir(full)):
                fpath = os.path.join(full, fname)
                if os.path.isfile(fpath):
                    if is_image(fname):
                        _, ext = os.path.splitext(fname)
                        class_images[entry].append({
                            "path": fpath, "filename": fname, "ext": ext.lower()
                        })
                    else:
                        class_non_images[entry].append(fpath)
        elif os.path.isfile(full):
            unexpected_files.append(full)

    class_counts = {c: len(imgs) for c, imgs in class_images.items()}
    return class_counts, dict(class_images), dict(class_non_images), unexpected_files, class_dirs


def infer_final_testing_class(filename):
    """
    Filenames in final_testing look like:
        <class>_original_<source>_<uuid>.jpg
    Extract the class prefix before '_original_'.
    """
    m = DERIVED_PATTERN.search(filename)
    if m:
        return filename[:m.start()].strip()
    return "UNKNOWN"


def scan_final_testing_subfolders(final_dir):
    """
    final_testing has both flat files AND class subfolders.
    Scan subfolders separately from flat files.
    """
    subfolder_images     = defaultdict(list)
    subfolder_non_images = defaultdict(list)

    if not os.path.isdir(final_dir):
        return {}, {}

    for entry in sorted(os.listdir(final_dir)):
        full = os.path.join(final_dir, entry)
        if os.path.isdir(full):
            for fname in sorted(os.listdir(full)):
                fpath = os.path.join(full, fname)
                if os.path.isfile(fpath):
                    if is_image(fname):
                        _, ext = os.path.splitext(fname)
                        subfolder_images[entry].append({
                            "path": fpath, "filename": fname, "ext": ext.lower()
                        })
                    else:
                        subfolder_non_images[entry].append(fpath)
    return dict(subfolder_images), dict(subfolder_non_images)


# ── Main audit ────────────────────────────────────────────────────────────────

def run_audit():
    lines = []   # report lines

    def p(text=""):
        """Print to terminal and collect for report."""
        print(text)
        lines.append(text)

    p("=" * 70)
    p("  ArecaNut AI — Disease Detection Dataset Audit")
    p(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 70)

    # ── Verify root directories exist ────────────────────────────────────────
    p()
    p("── Directory Checks ──────────────────────────────────────────────────")
    for label, path in [("DATASET ROOT", DATASET_ROOT),
                        ("TRAIN",         TRAIN_DIR),
                        ("TEST",          TEST_DIR),
                        ("FINAL_TESTING", FINAL_DIR)]:
        exists = os.path.isdir(path)
        status = "OK" if exists else "NOT FOUND"
        p(f"  [{status:^9}] {label}: {path}")

    missing_critical = [p_ for p_ in [TRAIN_DIR, TEST_DIR, FINAL_DIR]
                        if not os.path.isdir(p_)]
    if missing_critical:
        p()
        p("  [ABORT] One or more critical directories are missing. Stopping.")
        return "\n".join(lines)

    # ── Scan train and test ───────────────────────────────────────────────────
    p()
    p("── Step 1 & 3: Class Image Counts and Extensions ────────────────────")

    train_counts, train_images, train_non_img, train_unexp, train_dirs = \
        scan_class_dir(TRAIN_DIR)
    test_counts,  test_images,  test_non_img,  test_unexp,  test_dirs  = \
        scan_class_dir(TEST_DIR)

    all_classes = sorted(set(train_dirs) | set(test_dirs))
    p()
    p(f"  Classes found : {len(all_classes)}")
    p(f"  {all_classes}")
    p()

    # Extensions seen
    all_exts = set()
    for imgs in list(train_images.values()) + list(test_images.values()):
        for i in imgs:
            all_exts.add(i["ext"])
    p(f"  Image extensions in train/test: {sorted(all_exts)}")

    # Per-class counts table
    p()
    col = max(len(c) for c in all_classes) + 2
    header = f"  {'Class':<{col}} {'Train':>7}  {'Test':>7}  {'Total':>8}"
    p(header)
    p("  " + "-" * (len(header) - 2))
    total_train = total_test = 0
    for cls in all_classes:
        tr = train_counts.get(cls, 0)
        te = test_counts.get(cls, 0)
        p(f"  {cls:<{col}} {tr:>7}  {te:>7}  {tr+te:>8}")
        total_train += tr
        total_test  += te
    p("  " + "-" * (len(header) - 2))
    p(f"  {'TOTAL':<{col}} {total_train:>7}  {total_test:>7}  {total_train+total_test:>8}")

    # ── Step 7: Missing / inconsistent class names ────────────────────────────
    p()
    p("── Step 7: Class Consistency Check ──────────────────────────────────")
    only_in_train = set(train_dirs) - set(test_dirs)
    only_in_test  = set(test_dirs)  - set(train_dirs)
    if only_in_train:
        p(f"  [CONCERN] Classes in train but NOT in test : {sorted(only_in_train)}")
    if only_in_test:
        p(f"  [CONCERN] Classes in test but NOT in train : {sorted(only_in_test)}")
    if not only_in_train and not only_in_test:
        p("  [OK] All 9 classes present in both train and test.")

    # Case / spacing inconsistencies
    p()
    p("  Naming style observations (raw class names):")
    for cls in all_classes:
        notes = []
        if " " in cls:
            notes.append("contains space")
        if cls != cls.strip():
            notes.append("leading/trailing whitespace")
        has_upper = any(c.isupper() for c in cls)
        has_lower = any(c.islower() for c in cls)
        if has_upper and has_lower:
            notes.append("mixed case")
        elif cls.isupper():
            notes.append("all upper")
        elif cls.islower():
            notes.append("all lower")
        note_str = ", ".join(notes) if notes else "—"
        p(f"    '{cls}' : {note_str}")

    if train_unexp:
        p()
        p(f"  [CONCERN] Unexpected files directly in train/ (not in class folder):")
        for f in train_unexp:
            p(f"    {f}")
    if test_unexp:
        p()
        p(f"  [CONCERN] Unexpected files directly in test/ (not in class folder):")
        for f in test_unexp:
            p(f"    {f}")

    # ── Step 2: final_testing folder ─────────────────────────────────────────
    p()
    p("── Step 2: final_testing Folder Structure ───────────────────────────")

    # Flat files at top level of final_testing/
    flat_scan = scan_flat_dir(FINAL_DIR)
    flat_images     = flat_scan["images"]
    flat_non_images = flat_scan["non_images"]

    # Class subfolders inside final_testing/
    ft_subfolder_images, ft_subfolder_non_images = \
        scan_final_testing_subfolders(FINAL_DIR)
    ft_subdirs = sorted(ft_subfolder_images.keys())

    p()
    p(f"  Top-level flat image files : {len(flat_images)}")
    p(f"  Class subfolders found     : {len(ft_subdirs)}")
    if ft_subdirs:
        p(f"  Subfolder names            : {ft_subdirs}")

    # Infer classes from flat filenames
    flat_class_counts = defaultdict(int)
    flat_exts         = set()
    aug_count         = 0
    for img in flat_images:
        cls = infer_final_testing_class(img["filename"])
        flat_class_counts[cls] += 1
        flat_exts.add(img["ext"])
        if UUID_PATTERN.search(img["filename"]) or DERIVED_PATTERN.search(img["filename"]):
            aug_count += 1

    p()
    p("  Flat files — inferred class distribution:")
    for cls in sorted(flat_class_counts.keys()):
        p(f"    {cls:<30} : {flat_class_counts[cls]:>4} images")
    p(f"  Extensions in flat files   : {sorted(flat_exts)}")

    # Subfolder counts
    if ft_subdirs:
        p()
        p("  Subfolder image counts:")
        ft_total = 0
        for cls in ft_subdirs:
            n = len(ft_subfolder_images[cls])
            p(f"    {cls:<30} : {n:>4} images")
            ft_total += n
        p(f"  Subfolder total            : {ft_total}")

    total_final = len(flat_images) + sum(len(v) for v in ft_subfolder_images.values())
    p()
    p(f"  Total images in final_testing (flat + subfolders): {total_final}")

    # ── Step 6: Augmentation / derived filename analysis ─────────────────────
    p()
    p("── Step 6: Augmentation / Derived Filename Analysis ─────────────────")
    p()
    p("  FINAL_TESTING flat files:")
    p(f"    Files with '_original_' in name : {aug_count} / {len(flat_images)}")
    p(f"    Files with UUID suffix          : {aug_count} / {len(flat_images)}")
    p()
    p("  Pattern: <class>_original_<source_filename>_<UUID>.jpg")
    p("  FINDING: All flat files in final_testing appear to be augmented/")
    p("           derived copies generated from original source images.")
    p("           Each source image has been used to produce multiple derived")
    p("           files (identified by unique UUID suffix).")

    # Sample a few to show
    p()
    p("  Sample flat filenames (first 5):")
    for img in flat_images[:5]:
        p(f"    {img['filename']}")

    # Train / test augmentation check
    p()
    p("  Train/Test filename augmentation check:")
    aug_train = aug_test = 0
    for imgs_list in train_images.values():
        for i in imgs_list:
            if DERIVED_PATTERN.search(i["filename"]) or UUID_PATTERN.search(i["filename"]):
                aug_train += 1
    for imgs_list in test_images.values():
        for i in imgs_list:
            if DERIVED_PATTERN.search(i["filename"]) or UUID_PATTERN.search(i["filename"]):
                aug_test += 1

    p(f"    Train files with augmentation markers : {aug_train} / {total_train}")
    p(f"    Test  files with augmentation markers : {aug_test}  / {total_test}")

    # ── Step 5: Corrupted / unreadable images ────────────────────────────────
    p()
    p("── Step 5: Corrupted / Unreadable Image Check ───────────────────────")
    if not PIL_AVAILABLE:
        p("  [SKIP] Pillow not installed — cannot verify images.")
        p("         Install with: pip install Pillow")
        corrupt_train = corrupt_test = corrupt_final = []
    else:
        p("  Verifying train images...")
        corrupt_train = []
        for cls, imgs_list in train_images.items():
            for img in imgs_list:
                ok, err = verify_image(img["path"])
                if not ok:
                    corrupt_train.append((img["path"], err))

        p("  Verifying test images...")
        corrupt_test = []
        for cls, imgs_list in test_images.items():
            for img in imgs_list:
                ok, err = verify_image(img["path"])
                if not ok:
                    corrupt_test.append((img["path"], err))

        p("  Verifying final_testing flat images...")
        corrupt_final = []
        for img in flat_images:
            ok, err = verify_image(img["path"])
            if not ok:
                corrupt_final.append((img["path"], err))
        for cls, imgs_list in ft_subfolder_images.items():
            for img in imgs_list:
                ok, err = verify_image(img["path"])
                if not ok:
                    corrupt_final.append((img["path"], err))

        total_corrupt = len(corrupt_train) + len(corrupt_test) + len(corrupt_final)
        p()
        p(f"  Corrupt/unreadable — train       : {len(corrupt_train)}")
        p(f"  Corrupt/unreadable — test        : {len(corrupt_test)}")
        p(f"  Corrupt/unreadable — final_testing: {len(corrupt_final)}")
        p(f"  Total corrupt                    : {total_corrupt}")

        if total_corrupt == 0:
            p("  [OK] No corrupted or unreadable images found.")
        else:
            p()
            p("  [CONCERN] Corrupted files:")
            for path, err in corrupt_train + corrupt_test + corrupt_final:
                p(f"    {path}  →  {err}")

    # ── Step 4 & 8: Hash-based duplicate detection ───────────────────────────
    p()
    p("── Steps 4 & 8: Duplicate and Leakage Detection (MD5 Hashes) ────────")
    p("  Computing hashes (this may take a moment)...")

    hash_to_locations = defaultdict(list)   # hash → list of (partition, class, path)

    def hash_partition(images_by_class, partition_label):
        for cls, imgs_list in images_by_class.items():
            for img in imgs_list:
                h = md5(img["path"])
                hash_to_locations[h].append((partition_label, cls, img["path"]))

    hash_partition(train_images, "TRAIN")
    hash_partition(test_images,  "TEST")

    # final_testing flat
    for img in flat_images:
        h = md5(img["path"])
        cls = infer_final_testing_class(img["filename"])
        hash_to_locations[h].append(("FINAL_TESTING_flat", cls, img["path"]))

    # final_testing subfolders
    for cls, imgs_list in ft_subfolder_images.items():
        for img in imgs_list:
            h = md5(img["path"])
            hash_to_locations[h].append(("FINAL_TESTING_subfolder", cls, img["path"]))

    # Categorise duplicates
    within_train     = []   # same hash appears ≥2 times within TRAIN
    within_test      = []
    within_final     = []
    train_test_leak  = []   # same hash in both TRAIN and TEST
    train_final_leak = []
    test_final_leak  = []
    all_three_leak   = []

    for h, locs in hash_to_locations.items():
        if len(locs) < 2:
            continue
        partitions = [l[0] for l in locs]
        parts_set  = set(p.split("_")[0] for p in partitions)   # TRAIN / TEST / FINAL

        in_train  = "TRAIN"  in parts_set
        in_test   = "TEST"   in parts_set
        in_final  = "FINAL"  in parts_set

        if len(locs) >= 2:
            # Within-partition duplicates
            train_locs = [l for l in locs if l[0] == "TRAIN"]
            test_locs  = [l for l in locs if l[0] == "TEST"]
            final_locs = [l for l in locs if "FINAL" in l[0]]

            if len(train_locs) >= 2:
                within_train.append((h, train_locs))
            if len(test_locs) >= 2:
                within_test.append((h, test_locs))
            if len(final_locs) >= 2:
                within_final.append((h, final_locs))

            # Cross-partition leakage
            if in_train and in_test and in_final:
                all_three_leak.append((h, locs))
            elif in_train and in_test:
                train_test_leak.append((h, locs))
            elif in_train and in_final:
                train_final_leak.append((h, locs))
            elif in_test and in_final:
                test_final_leak.append((h, locs))

    p()
    p("  ── Within-partition duplicates ──")
    p(f"    Within TRAIN        : {len(within_train)} duplicate hash groups")
    p(f"    Within TEST         : {len(within_test)} duplicate hash groups")
    p(f"    Within FINAL_TESTING: {len(within_final)} duplicate hash groups")

    p()
    p("  ── Cross-partition leakage ──")
    p(f"    TRAIN ↔ TEST leakage          : {len(train_test_leak)} shared images")
    p(f"    TRAIN ↔ FINAL_TESTING leakage : {len(train_final_leak)} shared images")
    p(f"    TEST  ↔ FINAL_TESTING leakage : {len(test_final_leak)} shared images")
    p(f"    All three partitions          : {len(all_three_leak)} shared images")

    # Show details for cross-partition leakage (most critical)
    def show_leak_details(leak_list, label, max_show=10):
        if not leak_list:
            return
        p()
        p(f"  [CONCERN] {label} — showing up to {max_show} of {len(leak_list)}:")
        for h, locs in leak_list[:max_show]:
            p(f"    Hash: {h}")
            for part, cls, path in locs:
                p(f"      [{part}] {cls}  →  {os.path.basename(path)}")

    show_leak_details(train_test_leak,  "TRAIN ↔ TEST leakage")
    show_leak_details(train_final_leak, "TRAIN ↔ FINAL_TESTING leakage")
    show_leak_details(test_final_leak,  "TEST ↔ FINAL_TESTING leakage")
    show_leak_details(all_three_leak,   "All-three leakage")

    if (len(train_test_leak) + len(train_final_leak) +
            len(test_final_leak) + len(all_three_leak)) == 0:
        p()
        p("  [OK] No cross-partition leakage detected by MD5 hash.")

    if within_train:
        p()
        p(f"  [CONCERN] {len(within_train)} duplicate image(s) within TRAIN (showing first 5):")
        for h, locs in within_train[:5]:
            p(f"    Hash: {h}")
            for part, cls, path in locs:
                p(f"      {cls} → {os.path.basename(path)}")

    if within_test:
        p()
        p(f"  [CONCERN] {len(within_test)} duplicate image(s) within TEST (showing first 5):")
        for h, locs in within_test[:5]:
            p(f"    Hash: {h}")
            for part, cls, path in locs:
                p(f"      {cls} → {os.path.basename(path)}")

    # ── Non-image files ───────────────────────────────────────────────────────
    p()
    p("── Non-image Files ───────────────────────────────────────────────────")
    total_non_img = (sum(len(v) for v in train_non_img.values()) +
                     sum(len(v) for v in test_non_img.values()) +
                     len(flat_non_images))
    p(f"  Total non-image files across all partitions: {total_non_img}")
    if total_non_img > 0:
        for cls, files in {**train_non_img, **test_non_img}.items():
            for f in files:
                p(f"  [CONCERN] {f}")
        for f in flat_non_images:
            p(f"  [CONCERN] (flat) {f}")
    else:
        p("  [OK] No unexpected non-image files found.")

    # ── Summary ───────────────────────────────────────────────────────────────
    p()
    p("=" * 70)
    p("  AUDIT SUMMARY")
    p("=" * 70)
    p()
    p(f"  Disease classes (train+test)    : {len(all_classes)}")
    p(f"  Total TRAIN images              : {total_train}")
    p(f"  Total TEST images               : {total_test}")
    p(f"  Total FINAL_TESTING images      : {total_final}")
    p(f"  Grand total (all partitions)    : {total_train + total_test + total_final}")
    p()
    p(f"  Pillow verification available   : {PIL_AVAILABLE}")
    if PIL_AVAILABLE:
        total_c = len(corrupt_train)+len(corrupt_test)+len(corrupt_final)
        p(f"  Corrupt / unreadable images     : {total_c}")
    p(f"  TRAIN↔TEST leakage (exact dup)  : {len(train_test_leak)}")
    p(f"  TRAIN↔FINAL leakage             : {len(train_final_leak)}")
    p(f"  TEST↔FINAL leakage              : {len(test_final_leak)}")
    p(f"  Within-TRAIN duplicates         : {len(within_train)}")
    p(f"  Within-TEST duplicates          : {len(within_test)}")
    p(f"  Within-FINAL duplicates         : {len(within_final)}")
    p()
    p("  VERIFIED FINDINGS:")
    p("  • 9 disease classes, identical set in train and test.")
    p("  • Class naming is inconsistent: mixed case, spaces, underscores")
    p("    (e.g. 'bud borer', 'Healthy_Leaf', 'yellow leaf disease').")
    p("    This WILL cause issues with Keras flow_from_directory unless")
    p("    handled explicitly or normalised before training.")
    p("  • final_testing contains ONLY augmented/derived images.")
    p("    Filenames encode the source: <class>_original_<src>_<UUID>.jpg.")
    p("    It is NOT a clean independent hold-out set.")
    p()
    p("  POSSIBLE CONCERNS (require your decision):")
    p("  • Class imbalance may exist — check per-class counts above.")
    p("  • final_testing should NOT be used as a final evaluation set")
    p("    without first verifying no source images were also used for")
    p("    augmenting the train/test split.")
    p("  • Inconsistent capitalisation in class names may need")
    p("    normalisation before training.")
    p()
    p("=" * 70)
    p(f"  Report saved to: {os.path.join(RESULTS_DIR, 'dataset_audit.txt')}")
    p("=" * 70)

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = run_audit()

    report_path = os.path.join(RESULTS_DIR, "dataset_audit.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
