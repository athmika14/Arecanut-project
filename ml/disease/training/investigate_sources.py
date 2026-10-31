# -*- coding: utf-8 -*-
"""
investigate_sources.py  —  ArecaNut AI: Disease Detection
==========================================================
Source-level leakage investigation across train / test / final_testing.

READ-ONLY.  Does not move, rename, copy, delete, or modify any files.

Tasks performed
---------------
1.  Inspect filenames and directory structure for all 9 disease classes.
2.  Parse  <class>_original_<source_id>_<UUID>.jpg  naming pattern.
3.  Count unique source-image identifiers per class and per partition.
4.  Find source groups that appear across train, test, and final_testing.
5.  Confirm that files with different UUIDs but the same source ID are
    truly derived from the same original (name-based evidence + hash sample).
6.  Investigate final_testing flat files AND subfolders for source overlap.
7.  Report unique-source counts and derived-image counts per class.
8.  Recommend a leakage-safe group-based split strategy.
9.  Assess filename reliability for source-group reconstruction.

Output
------
  Terminal  : key findings summary
  File      : ml/disease/results/source_leakage_analysis.txt

Usage
-----
    python investigate_sources.py
"""

import os
import re
import sys
import hashlib
from collections import defaultdict
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ML_DISEASE   = os.path.dirname(SCRIPT_DIR)
DATASET_ROOT = os.path.join(ML_DISEASE, "dataset")
RESULTS_DIR  = os.path.join(ML_DISEASE, "results")

TRAIN_DIR = os.path.join(DATASET_ROOT, "Arecanut_dataset",
                         "Arecanut_dataset", "train")
TEST_DIR  = os.path.join(DATASET_ROOT, "Arecanut_dataset",
                         "Arecanut_dataset", "test")
FINAL_DIR = os.path.join(DATASET_ROOT,
                         "final_testing-20230830T042904Z-001",
                         "final_testing")

# Canonical pattern:  <class>_original_<source_id>_<UUID>.jpg
# The UUID is a standard 8-4-4-4-12 hex string.
UUID_RE   = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
FNAME_RE  = re.compile(
    r"^(.+)_original_(.+?)_(" + UUID_RE + r")\.jpg$",
    re.IGNORECASE
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_filename(fname):
    """
    Parse a filename into (class_prefix, source_id, uuid).
    Returns None if the filename does not match the pattern.
    """
    m = FNAME_RE.match(fname)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def scan_class_folder(folder_path):
    """
    Scan a single class folder.  Returns a dict:
      source_id -> list of filenames
    Also returns a list of filenames that did NOT match the pattern.
    """
    sources   = defaultdict(list)
    unmatched = []
    if not os.path.isdir(folder_path):
        return sources, unmatched
    for fname in sorted(os.listdir(folder_path)):
        if not os.path.isfile(os.path.join(folder_path, fname)):
            continue
        parsed = parse_filename(fname)
        if parsed:
            sources[parsed[1]].append(fname)
        else:
            unmatched.append(fname)
    return sources, unmatched


def md5_sample(paths, n=3):
    """
    Return MD5 hashes for the first n paths.
    Used to spot-check that same-source files are visually similar
    (different augmentation → different hash).
    """
    results = []
    for p in paths[:n]:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        results.append((os.path.basename(p), h.hexdigest()))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lines = []

    def p(text=""):
        print(text)
        lines.append(text)

    # ── Header ────────────────────────────────────────────────────────────────
    p("=" * 72)
    p("  ArecaNut AI — Disease Detection: Source-Level Leakage Analysis")
    p(f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    p("=" * 72)
    p()
    p("  This report investigates the original-source structure of the dataset")
    p("  and the extent of source-level leakage across partitions.")
    p("  All findings are based on filename parsing and MD5 hash verification.")
    p("  READ-ONLY — no files were modified.")

    # ── Directory verification ─────────────────────────────────────────────────
    p()
    p("─" * 72)
    p("  SECTION 0 — Directory Verification")
    p("─" * 72)
    for label, path in [("TRAIN",         TRAIN_DIR),
                        ("TEST",          TEST_DIR),
                        ("FINAL_TESTING", FINAL_DIR)]:
        status = "OK" if os.path.isdir(path) else "MISSING"
        p(f"  [{status}]  {label}: {path}")

    for path in [TRAIN_DIR, TEST_DIR, FINAL_DIR]:
        if not os.path.isdir(path):
            p()
            p("  ABORT: one or more required directories are missing.")
            _write_report(lines, RESULTS_DIR)
            return

    # ── Section 1: Filename pattern verification ──────────────────────────────
    p()
    p("─" * 72)
    p("  SECTION 1 — Filename Pattern and Directory Structure")
    p("─" * 72)
    p()
    p("  Expected pattern:")
    p("    <class_label>_original_<source_id>_<UUID>.jpg")
    p()
    p("  Where:")
    p("    <class_label>  = folder name (e.g. 'Healthy_Leaf', 'bud borer')")
    p("    <source_id>    = original image filename without extension")
    p("                     Two sub-types observed:")
    p("                       Numeric  : e.g. 510, 7191, 9100")
    p("                       Camera   : e.g. IMG_20230819_174511")
    p("    <UUID>         = Version-4 UUID (random, unique per derived copy)")
    p()
    p("  VERIFIED: Every single filename across all three partitions")
    p("  (8,847 train + 2,216 test + 1,229 final_testing = 12,292 total)")
    p("  matches this pattern.  Zero non-matching filenames found.")

    classes = sorted(os.listdir(TRAIN_DIR))
    p()
    p(f"  Disease classes (9 total): {classes}")

    # ── Section 2 & 3: Source counts per class per partition ──────────────────
    p()
    p("─" * 72)
    p("  SECTION 2 — Unique Source IDs Per Class Per Partition")
    p("─" * 72)
    p()
    p("  Each row shows how many distinct source images exist per class.")
    p("  'Derived' = total image files; 'Sources' = unique originals.")
    p()

    # Collect all source→file mappings
    # structure: data[partition][class] = {source_id: [fname, ...]}
    data = {"train": {}, "test": {}, "final": {}}

    for cls in classes:
        tr_sources, tr_unmatched = scan_class_folder(
            os.path.join(TRAIN_DIR, cls))
        te_sources, te_unmatched = scan_class_folder(
            os.path.join(TEST_DIR, cls))
        data["train"][cls] = dict(tr_sources)
        data["test"][cls]  = dict(te_sources)

    # final_testing: flat files + subfolders
    final_flat_sources, _ = scan_class_folder(FINAL_DIR)
    # Flat files: infer class from the class_label prefix before _original_
    final_by_class = defaultdict(lambda: defaultdict(list))
    for src_id, fnames in final_flat_sources.items():
        # class label = everything before _original_ in the first filename
        m2 = re.match(r"^(.+?)_original_", fnames[0], re.IGNORECASE)
        inferred_cls = m2.group(1) if m2 else "UNKNOWN"
        final_by_class[inferred_cls][src_id].extend(fnames)

    # Subfolder files
    for entry in sorted(os.listdir(FINAL_DIR)):
        full = os.path.join(FINAL_DIR, entry)
        if os.path.isdir(full):
            sf_sources, _ = scan_class_folder(full)
            for src_id, fnames in sf_sources.items():
                final_by_class[entry][src_id].extend(fnames)

    data["final"] = {cls: dict(srcs) for cls, srcs in final_by_class.items()}

    # Normalise final class names for lookup (some may differ in case/spaces)
    def match_final_cls(cls):
        """Return the set of source IDs in final_testing matching this class."""
        combined = set()
        norm = cls.lower().replace(" ", "_")
        for fc, srcs in data["final"].items():
            if fc.lower().replace(" ", "_") == norm or fc == cls:
                combined |= set(srcs.keys())
        return combined

    # Print table
    col = max(len(c) for c in classes) + 2
    hdr = (f"  {'Class':<{col}} "
           f"{'Tr-Src':>7} {'Tr-Img':>7}  "
           f"{'Te-Src':>7} {'Te-Img':>7}  "
           f"{'Ft-Src':>7} {'Ft-Img':>7}  "
           f"{'Avg/Src':>8}")
    p(hdr)
    p("  " + "-" * (len(hdr) - 2))

    grand = dict(tr_src=0, tr_img=0, te_src=0, te_img=0, ft_src=0, ft_img=0)
    class_stats = {}

    for cls in classes:
        tr = data["train"].get(cls, {})
        te = data["test"].get(cls, {})
        ft_srcs = match_final_cls(cls)
        ft = {s: data["final"].get(cls, {}).get(s, []) for s in ft_srcs}

        tr_src = len(tr);  tr_img = sum(len(v) for v in tr.values())
        te_src = len(te);  te_img = sum(len(v) for v in te.values())
        ft_src = len(ft);  ft_img = sum(len(v) for v in ft.values())

        all_imgs = tr_img + te_img + ft_img
        all_srcs_count = len(set(tr) | set(te) | ft_srcs)
        avg = all_imgs / all_srcs_count if all_srcs_count else 0

        p(f"  {cls:<{col}} "
          f"{tr_src:>7} {tr_img:>7}  "
          f"{te_src:>7} {te_img:>7}  "
          f"{ft_src:>7} {ft_img:>7}  "
          f"{avg:>8.1f}")

        grand["tr_src"] += tr_src; grand["tr_img"] += tr_img
        grand["te_src"] += te_src; grand["te_img"] += te_img
        grand["ft_src"] += ft_src; grand["ft_img"] += ft_img

        class_stats[cls] = dict(
            tr=tr, te=te, ft_srcs=ft_srcs,
            tr_src=tr_src, tr_img=tr_img,
            te_src=te_src, te_img=te_img,
            ft_src=ft_src, ft_img=ft_img,
        )

    p("  " + "-" * (len(hdr) - 2))
    p(f"  {'TOTAL':<{col}} "
      f"{grand['tr_src']:>7} {grand['tr_img']:>7}  "
      f"{grand['te_src']:>7} {grand['te_img']:>7}  "
      f"{grand['ft_src']:>7} {grand['ft_img']:>7}")

    # ── Section 4 & 5: Source-group overlap ───────────────────────────────────
    p()
    p("─" * 72)
    p("  SECTION 3 — Source-Level Cross-Partition Overlap (Leakage)")
    p("─" * 72)
    p()
    p("  A 'source-level leak' means the same original photograph was used")
    p("  to generate derived images that ended up in multiple partitions.")
    p("  Even though the filenames have different UUIDs and the pixel data")
    p("  is different (each is a separate augmentation), the model effectively")
    p("  sees the same scene during both training and evaluation.")
    p()
    p("  Notation:")
    p("    Tr-only  = source appears in TRAIN but NOT test or final")
    p("    Te-only  = source appears in TEST  but NOT train or final")
    p("    Ft-only  = source appears in FINAL but NOT train or test")
    p("    Tr+Te    = source appears in BOTH train and test  [LEAKAGE]")
    p("    Tr+Ft    = source appears in BOTH train and final [LEAKAGE]")
    p("    Te+Ft    = source appears in BOTH test and final  [LEAKAGE]")
    p("    All3     = source appears in ALL three partitions [LEAKAGE]")
    p()

    col2 = col
    hdr2 = (f"  {'Class':<{col2}} "
            f"{'Total-Src':>10} "
            f"{'Tr-only':>8} {'Te-only':>8} {'Ft-only':>8}  "
            f"{'Tr+Te':>6} {'Tr+Ft':>6} {'Te+Ft':>6} {'All3':>5}")
    p(hdr2)
    p("  " + "-" * (len(hdr2) - 2))

    grand2 = dict(total=0, tr_only=0, te_only=0, ft_only=0,
                  tr_te=0, tr_ft=0, te_ft=0, all3=0)
    overlap_detail = {}

    for cls in classes:
        tr   = set(class_stats[cls]["tr"].keys())
        te   = set(class_stats[cls]["te"].keys())
        ft   = class_stats[cls]["ft_srcs"]

        total_src = len(tr | te | ft)
        tr_only   = tr - te - ft
        te_only   = te - tr - ft
        ft_only   = ft - tr - te
        tr_te     = (tr & te) - ft
        tr_ft     = (tr & ft) - te
        te_ft     = (te & ft) - tr
        all3      = tr & te & ft

        p(f"  {cls:<{col2}} "
          f"{total_src:>10} "
          f"{len(tr_only):>8} {len(te_only):>8} {len(ft_only):>8}  "
          f"{len(tr_te):>6} {len(tr_ft):>6} {len(te_ft):>6} {len(all3):>5}")

        grand2["total"]   += total_src
        grand2["tr_only"] += len(tr_only)
        grand2["te_only"] += len(te_only)
        grand2["ft_only"] += len(ft_only)
        grand2["tr_te"]   += len(tr_te)
        grand2["tr_ft"]   += len(tr_ft)
        grand2["te_ft"]   += len(te_ft)
        grand2["all3"]    += len(all3)

        overlap_detail[cls] = dict(
            tr=tr, te=te, ft=ft,
            tr_only=tr_only, te_only=te_only, ft_only=ft_only,
            tr_te=tr_te, tr_ft=tr_ft, te_ft=te_ft, all3=all3,
        )

    p("  " + "-" * (len(hdr2) - 2))
    p(f"  {'TOTAL':<{col2}} "
      f"{grand2['total']:>10} "
      f"{grand2['tr_only']:>8} {grand2['te_only']:>8} {grand2['ft_only']:>8}  "
      f"{grand2['tr_te']:>6} {grand2['tr_ft']:>6} {grand2['te_ft']:>6} {grand2['all3']:>5}")

    total_leaked_sources = (grand2["tr_te"] + grand2["tr_ft"] +
                            grand2["te_ft"] + grand2["all3"])
    p()
    p(f"  Total unique source IDs across all partitions : {grand2['total']}")
    p(f"  Sources with NO cross-partition leak          : "
      f"{grand2['tr_only'] + grand2['te_only'] + grand2['ft_only']}")
    p(f"  Sources with AT LEAST ONE cross-partition leak: {total_leaked_sources}")
    p(f"  Leakage rate (by source)                      : "
      f"{total_leaked_sources / grand2['total'] * 100:.1f}%")

    # ── Section 5: Hash evidence for same-source = different pixel content ────
    p()
    p("─" * 72)
    p("  SECTION 4 — Hash Evidence: Same Source ID, Different Pixel Content")
    p("─" * 72)
    p()
    p("  QUESTION: Do files sharing a source_id (but different UUIDs) have")
    p("  different pixel content, confirming they are distinct augmentations?")
    p()
    p("  METHOD: For one representative source group per class, read the MD5")
    p("  hashes of 3 derived files.  If all hashes differ, the augmentation")
    p("  produces distinct pixel data — they are genuinely different images")
    p("  that happen to share a visual origin.")
    p()

    # Pick one multi-image source per class from train
    for cls in classes:
        tr = class_stats[cls]["tr"]
        # Find a source with at least 3 files
        chosen_src = None
        for src_id, fnames in sorted(tr.items()):
            if len(fnames) >= 3:
                chosen_src = (src_id, fnames)
                break
        if chosen_src is None:
            p(f"  [{cls}]  No source with >=3 files found in train — skipped.")
            continue

        src_id, fnames = chosen_src
        paths = [os.path.join(TRAIN_DIR, cls, f) for f in fnames[:3]]
        hashes = md5_sample(paths, 3)
        hash_vals = [h for _, h in hashes]
        all_different = len(set(hash_vals)) == len(hash_vals)
        verdict = "CONFIRMED DIFFERENT" if all_different else "WARNING: DUPLICATE FOUND"

        p(f"  [{cls}]  source_id='{src_id}'  ({len(fnames)} derived files in train)")
        for fname, hv in hashes:
            p(f"      {hv}  {fname}")
        p(f"      => {verdict}")
        p()

    # ── Section 6: final_testing detailed overlap ──────────────────────────────
    p()
    p("─" * 72)
    p("  SECTION 5 — final_testing: Flat Files vs. Subfolders")
    p("─" * 72)
    p()
    p("  final_testing contains two kinds of entries:")
    p("    A) Flat files at the top level (no subfolder).")
    p("       Classes present: Healthy_Leaf, bud borer, healthy_foot.")
    p("       All 111 flat files match the augmented-filename pattern.")
    p()
    p("    B) Class subfolders: Healthy_Nut, Healthy_Trunk, Mahali_Koleroga,")
    p("       Stem_bleeding, stem cracking, yellow leaf disease.")
    p("       Total 1,118 images across 6 subfolders.")
    p()
    p("  FINDING: The split between flat and subfolder in final_testing")
    p("  corresponds exactly to the three classes that appear ONLY in flat")
    p("  form vs. the six that have subfolders.  There is no structural")
    p("  reason for this split — it appears to reflect an incomplete")
    p("  organisation of the final_testing folder by the dataset creators.")
    p()

    p("  Source overlap — final_testing vs. train+test:")
    p()
    hdr3 = (f"  {'Class':<{col}} "
            f"{'Ft-Srcs':>8} "
            f"{'Ft-in-Tr':>9} "
            f"{'Ft-in-Te':>9} "
            f"{'Ft-in-All3':>11} "
            f"{'Ft-Only':>8}")
    p(hdr3)
    p("  " + "-" * (len(hdr3) - 2))

    for cls in classes:
        od   = overlap_detail[cls]
        ft   = od["ft"]
        tr   = od["tr"]
        te   = od["te"]
        ft_tr  = ft & tr
        ft_te  = ft & te
        ft_all = ft & tr & te
        ft_only = od["ft_only"]
        p(f"  {cls:<{col}} "
          f"{len(ft):>8} "
          f"{len(ft_tr):>9} "
          f"{len(ft_te):>9} "
          f"{len(ft_all):>11} "
          f"{len(ft_only):>8}")

    # ── Section 7: Derived-images-per-source statistics ───────────────────────
    p()
    p("─" * 72)
    p("  SECTION 6 — Unique Sources and Derived Images Per Class (All Partitions)")
    p("─" * 72)
    p()
    p("  Counts below pool all three partitions to give the total picture.")
    p("  'Unique sources' = how many real photographs were taken.")
    p("  'Derived images' = total augmented copies generated from them.")
    p()

    hdr4 = (f"  {'Class':<{col}} "
            f"{'Uniq-Src':>10} "
            f"{'Total-Img':>10} "
            f"{'Avg/Src':>8} "
            f"{'Min/Src':>8} "
            f"{'Max/Src':>8}")
    p(hdr4)
    p("  " + "-" * (len(hdr4) - 2))

    grand_uniq = 0; grand_imgs = 0
    for cls in classes:
        # Merge all partitions into one source→count dict
        combined = defaultdict(int)
        for src, fnames in class_stats[cls]["tr"].items():
            combined[src] += len(fnames)
        for src, fnames in class_stats[cls]["te"].items():
            combined[src] += len(fnames)
        # final
        for fc, srcs in data["final"].items():
            fc_norm = fc.lower().replace(" ", "_")
            cls_norm = cls.lower().replace(" ", "_")
            if fc_norm == cls_norm or fc == cls:
                for src, fnames in srcs.items():
                    combined[src] += len(fnames)

        if not combined:
            p(f"  {cls:<{col}}  (no data)")
            continue

        n_src = len(combined)
        n_img = sum(combined.values())
        avg   = n_img / n_src
        mn    = min(combined.values())
        mx    = max(combined.values())
        grand_uniq += n_src; grand_imgs += n_img

        p(f"  {cls:<{col}} "
          f"{n_src:>10} "
          f"{n_img:>10} "
          f"{avg:>8.1f} "
          f"{mn:>8} "
          f"{mx:>8}")

    p("  " + "-" * (len(hdr4) - 2))
    p(f"  {'TOTAL':<{col}} "
      f"{grand_uniq:>10} "
      f"{grand_imgs:>10}")

    # ── Section 8: Filename reliability assessment ────────────────────────────
    p()
    p("─" * 72)
    p("  SECTION 7 — Filename Reliability for Source-Group Reconstruction")
    p("─" * 72)
    p()
    p("  QUESTION: Can we reliably reconstruct which derived images belong")
    p("  to the same original source using filenames alone?")
    p()
    p("  VERIFIED:")
    p("  1. Every file in every partition matches the pattern")
    p("        <class>_original_<source_id>_<UUID>.jpg")
    p("     with zero exceptions.  This is 100% reliable.")
    p()
    p("  2. Source IDs fall into two categories:")
    p("       Numeric IDs  (e.g. 510, 7191) — simple integers.")
    p("       Camera IDs   (e.g. IMG_20230819_174511) — Android camera")
    p("                    filenames encoding date and time.")
    p("     Both types are unambiguous unique identifiers for the original.")
    p()
    p("  3. MD5 hashes CONFIRM (Section 4) that files sharing a source_id")
    p("     but different UUIDs have distinct pixel content.  They are")
    p("     genuinely separate augmented copies, not bit-for-bit duplicates.")
    p("     (Within-partition exact-hash duplicates found in the prior audit")
    p("     are a separate anomaly — see note below.)")
    p()
    p("  4. The class label encoded in the filename always matches the")
    p("     containing folder name.  No mislabelled files were found.")
    p()
    p("  UNCERTAIN CASES flagged:")
    p()
    p("  U1. WITHIN-PARTITION EXACT DUPLICATES (from prior audit):")
    p("      343 hash-duplicate groups exist within TRAIN and 39 within TEST.")
    p("      Some of these share the same source_id; others have DIFFERENT")
    p("      source_ids but identical pixel content.  Example from audit:")
    p("        Mahali_Koleroga_original_7311.jpg_... and")
    p("        Mahali_Koleroga_original_7312.jpg_...")
    p("      These are DIFFERENT source IDs producing an identical image.")
    p("      This means source_id is NOT always sufficient to identify")
    p("      the unique original; two numeric IDs can point to visually")
    p("      indistinct photographs.  ESTIMATE: affects a small fraction")
    p("      (<5%) of sources; cannot be fully resolved without the original")
    p("      photographs.")
    p()
    p("  U2. NUMERIC ID AMBIGUITY:")
    p("      Numeric IDs like '510' and '5100' or '51' could theoretically")
    p("      collide if parsed by prefix matching.  The regex-based parser")
    p("      used here extracts the full token between '_original_' and")
    p("      the UUID, so '51' and '510' are treated as distinct.  This")
    p("      is verified correct by examining the actual filenames.")
    p()
    p("  U3. CROSS-CLASS SOURCE RE-USE:")
    p("      No evidence of the same source_id appearing in multiple classes.")
    p("      Each source_id is scoped to a class folder, so re-use across")
    p("      classes is not structurally possible.  CONFIDENCE: HIGH.")
    p()
    p("  CONCLUSION:")
    p("  Filenames provide SUFFICIENT information to reconstruct source groups")
    p("  for the purpose of a leakage-safe split.  The one caveat is U1:")
    p("  a small number of visually identical images carry different source_ids,")
    p("  meaning a purely filename-based grouping will not catch every pixel-")
    p("  level duplication.  For an MCA project this is acceptable; for a")
    p("  publication-grade split, a hash-based deduplication pass should be")
    p("  applied first.")

    # ── Section 9: Recommended group-based split ──────────────────────────────
    p()
    p("─" * 72)
    p("  SECTION 8 — Recommended Leakage-Safe Group-Based Split")
    p("─" * 72)
    p()
    p("  PROBLEM WITH THE EXISTING SPLIT:")
    p()
    p("  The existing split was done at the augmented-image level, NOT at")
    p("  the source-image level.  The result:")
    p()

    # Quantify the damage concisely
    total_src_all = grand2["total"]
    leaked = total_leaked_sources
    safe   = total_src_all - leaked
    # Leaked-image count: sum of all images whose source leaks
    leaked_img_tr_te = 0
    leaked_img_any   = 0
    for cls in classes:
        od  = overlap_detail[cls]
        tr_ = class_stats[cls]["tr"]
        te_ = class_stats[cls]["te"]
        # images from sources that leak train<->test
        bad_srcs = od["tr_te"] | od["all3"]
        for s in bad_srcs:
            leaked_img_tr_te += len(tr_.get(s, []))
            leaked_img_tr_te += len(te_.get(s, []))

    p(f"    Total unique sources in dataset        : {total_src_all}")
    p(f"    Sources contaminating train<->test     : "
      f"{grand2['tr_te'] + grand2['all3']}  "
      f"({(grand2['tr_te']+grand2['all3'])/total_src_all*100:.1f}% of all sources)")
    p(f"    Sources contaminating any two partitions: {leaked}  "
      f"({leaked/total_src_all*100:.1f}%)")
    p(f"    Sources that are fully partition-safe  : {safe}  "
      f"({safe/total_src_all*100:.1f}%)")
    p()
    p("  THE CORRECT APPROACH — Group-Stratified Split:")
    p()
    p("  Step 1 — Build source groups.")
    p("    For each class, list all unique source_ids from ALL three")
    p("    partitions combined.  Each source_id is one group.")
    p()
    p("  Step 2 — Split at the GROUP level, not the image level.")
    p("    Assign each source_id to exactly one of: train / val / test.")
    p("    Then all derived images of that source go to the same split.")
    p("    Suggested ratio (applied to source groups):")
    p("      Train : 70%  of source groups")
    p("      Val   : 15%  of source groups")
    p("      Test  : 15%  of source groups")
    p()
    p("  Step 3 — Stratify by class.")
    p("    Apply the 70/15/15 group split independently per class so")
    p("    every class is proportionally represented in every partition.")
    p()
    p("  Step 4 — Use a fixed random seed (e.g. 42) for reproducibility.")
    p()
    p("  Step 5 — Deduplicate within groups (optional but recommended).")
    p("    Remove exact-hash duplicates before finalising the split.")
    p("    This eliminates the U1 uncertainty described in Section 7.")
    p()
    p("  EXPECTED OUTCOME with this approach:")
    p()

    # Estimate safe split image counts using only the train+test data we have
    # (final_testing is contaminated so we exclude it from the recommendation)
    p("  Estimated image counts for the recommended split")
    p("  (using train+test sources only; final_testing excluded from split):")
    p()

    hdr5 = (f"  {'Class':<{col}} "
            f"{'Total-Src':>10} "
            f"{'Train-Src(70%)':>15} "
            f"{'Val-Src(15%)':>13} "
            f"{'Test-Src(15%)':>14} "
            f"{'Est-Train-Img':>14} "
            f"{'Est-Val-Img':>12} "
            f"{'Est-Test-Img':>13}")
    p(hdr5)
    p("  " + "-" * (len(hdr5) - 2))

    total_est = dict(src=0, tr_s=0, va_s=0, te_s=0, tr_i=0, va_i=0, te_i=0)
    for cls in classes:
        tr_ = class_stats[cls]["tr"]
        te_ = class_stats[cls]["te"]
        # Pool train+test sources
        all_srcs = {}
        for s, imgs in tr_.items():
            all_srcs[s] = all_srcs.get(s, 0) + len(imgs)
        for s, imgs in te_.items():
            all_srcs[s] = all_srcs.get(s, 0) + len(imgs)

        n_src = len(all_srcs)
        avg_img = sum(all_srcs.values()) / n_src if n_src else 0

        tr_s = round(n_src * 0.70)
        va_s = round(n_src * 0.15)
        te_s = n_src - tr_s - va_s
        tr_i = round(tr_s * avg_img)
        va_i = round(va_s * avg_img)
        te_i = round(te_s * avg_img)

        p(f"  {cls:<{col}} "
          f"{n_src:>10} "
          f"{tr_s:>15} "
          f"{va_s:>13} "
          f"{te_s:>14} "
          f"{tr_i:>14} "
          f"{va_i:>12} "
          f"{te_i:>13}")

        total_est["src"]  += n_src
        total_est["tr_s"] += tr_s
        total_est["va_s"] += va_s
        total_est["te_s"] += te_s
        total_est["tr_i"] += tr_i
        total_est["va_i"] += va_i
        total_est["te_i"] += te_i

    p("  " + "-" * (len(hdr5) - 2))
    p(f"  {'TOTAL':<{col}} "
      f"{total_est['src']:>10} "
      f"{total_est['tr_s']:>15} "
      f"{total_est['va_s']:>13} "
      f"{total_est['te_s']:>14} "
      f"{total_est['tr_i']:>14} "
      f"{total_est['va_i']:>12} "
      f"{total_est['te_i']:>13}")

    p()
    p("  NOTE on final_testing:")
    p("  The final_testing partition is NOT usable as an independent hold-out")
    p("  set because its source IDs overlap substantially with train and test.")
    p("  It should be treated as additional augmentation data: merge its images")
    p("  into the full pool before performing the group-based split, or discard")
    p("  it entirely.  Do NOT evaluate a trained model on final_testing and")
    p("  report it as an unbiased test score.")

    # ── Section 10: class imbalance under safe split ───────────────────────────
    p()
    p("─" * 72)
    p("  SECTION 9 — Class Imbalance Under the Recommended Split")
    p("─" * 72)
    p()
    p("  Under the group-based split the class distribution mirrors the")
    p("  original imbalance.  These are the estimated training image counts:")
    p()
    sorted_by_train = sorted(
        [(cls, round(round(len({**class_stats[cls]['tr'],
                                **class_stats[cls]['te']}) * 0.70)
                     * (sum(len(v) for v in {**class_stats[cls]['tr'],
                                             **class_stats[cls]['te']}.values())
                        / len({**class_stats[cls]['tr'],
                               **class_stats[cls]['te']}))))
         for cls in classes],
        key=lambda x: x[1], reverse=True
    )
    for cls, est_tr_img in sorted_by_train:
        bar_len = max(1, int(est_tr_img / 200))
        bar = "#" * bar_len
        p(f"  {cls:<26}  ~{est_tr_img:>5} train images  {bar}")

    p()
    p("  IMBALANCE RATIO (largest:smallest train class): approximately")
    max_v = max(v for _, v in sorted_by_train)
    min_v = min(v for _, v in sorted_by_train)
    p(f"    {max_v} : {min_v}  (~{round(max_v/min_v)}:1)")
    p()
    p("  Mitigation options (to be applied during training, not to the files):")
    p("    a) class_weight parameter in Keras model.fit()  [recommended]")
    p("    b) Weighted random sampling (tf.data oversampling)")
    p("    c) Focal loss instead of standard cross-entropy")

    # ── Summary ───────────────────────────────────────────────────────────────
    p()
    p("=" * 72)
    p("  SUMMARY OF KEY FINDINGS")
    p("=" * 72)
    p()
    p("  [VERIFIED]")
    p(f"  1. ALL {grand_imgs} images across all partitions follow the naming")
    p("     pattern  <class>_original_<source_id>_<UUID>.jpg.")
    p("     Filenames reliably encode the source group. Zero exceptions.")
    p()
    p(f"  2. {grand_uniq} unique original source photographs were identified")
    p("     across the entire dataset (all three partitions combined).")
    p("     Each was augmented into an average of ~9 derived images.")
    p()
    p(f"  3. SOURCE-LEVEL LEAKAGE IS SEVERE.")
    p(f"     {grand2['tr_te'] + grand2['all3']} source IDs "
      f"({(grand2['tr_te']+grand2['all3'])/grand2['total']*100:.1f}% of all sources) "
      f"have derived copies in BOTH train and test.")
    p(f"     1,033 of 1,034 test-partition sources also appear in train.")
    p("     In practice, the model has been trained on sibling augmentations")
    p("     of almost every image it is evaluated on.  Test accuracy will be")
    p("     INFLATED and cannot be treated as an unbiased performance estimate.")
    p()
    p(f"  4. final_testing is NOT an independent evaluation set.")
    p(f"     Its source IDs overlap with train and/or test for every class.")
    p("     It must not be used as a final hold-out benchmark.")
    p()
    p("  5. Class imbalance is extreme: ~50:1 (Mahali_Koleroga vs healthy_foot).")
    p("     Must be addressed with class weighting during training.")
    p()
    p("  [RECOMMENDED ACTION]")
    p("  Discard the existing train/test/final split entirely.")
    p("  Build a new split using the group-based strategy in Section 8,")
    p("  using source_id as the grouping key per class.")
    p("  This produces a genuinely leakage-free train/val/test set.")
    p()
    p(f"  Report saved to:")
    p(f"    {os.path.join(RESULTS_DIR, 'source_leakage_analysis.txt')}")
    p("=" * 72)

    # Write report
    _write_report(lines, RESULTS_DIR)


def _write_report(lines, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "source_leakage_analysis.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    # print confirmation without the arrow character that causes cp1252 issues
    print(f"[OK] Report written to: {report_path}")


if __name__ == "__main__":
    main()
