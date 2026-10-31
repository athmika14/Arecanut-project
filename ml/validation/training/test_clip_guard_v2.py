# -*- coding: utf-8 -*-
"""
test_clip_guard_v2.py - CLIP Zero-Shot Validation Test (v2)
============================================================
Fixes from v1:
  - Removed "banana tree" and "coconut tree" from reject prompts
    (they are palms, too similar to arecanut - caused false rejections)
  - Added "a photo of a palm tree with disease symptoms" to arecanut prompts
  - Added "a photo of a screenshot" and "a photo of a keyboard" to non-plant
  - Added minimum image dimension check (50x50 pixels)
  - Tests 3 images per class (27 total) for more robust sampling
  - More synthetic controls (skin-tone, sky-blue, wood-brown)

Standalone test - does NOT modify any project files.
"""

import os, sys, io, warnings, traceback
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.dirname(SCRIPT_DIR)
ML_DIR         = os.path.dirname(VALIDATION_DIR)
DATASET_TEST_DIR = os.path.join(
    ML_DIR, "disease", "dataset", "Arecanut_dataset",
    "Arecanut_dataset", "test"
)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ── Prompt design (v2) ────────────────────────────────────────────────────────
# Key change: banana tree and coconut tree REMOVED from reject prompts.
# They are palm species visually similar to arecanut. Including them
# caused false rejections of Stem_bleeding images.
# CLIP cannot distinguish arecanut from other palms - this is a known
# limitation that must be documented, not hidden.

ARECANUT_PROMPTS = [
    "a photo of an arecanut palm tree",
    "a photo of areca nuts on a palm tree",
    "a photo of areca nuts",
    "a photo of betel nuts",
    "a photo of a palm tree trunk",
    "a photo of a palm leaf",
    "a photo of a diseased palm leaf",
    "a photo of a tropical palm plantation",
    "a photo of a palm tree stem",
    "a photo of a palm tree with disease symptoms",
]

NON_PALM_PLANT_PROMPTS = [
    "a photo of a rose flower",
    "a photo of a tomato plant",
    "a photo of a wheat field",
    "a photo of a cactus",
    "a photo of a sunflower",
    "a photo of a potted houseplant",
]

NON_PLANT_PROMPTS = [
    "a photo of a person",
    "a photo of a baby",
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a car",
    "a photo of a building",
    "a photo of a phone",
    "a photo of food on a plate",
    "a photo of a landscape",
    "a photo of a blank image",
    "a photo of a screenshot",
    "a photo of a keyboard",
]

ALL_PROMPTS = ARECANUT_PROMPTS + NON_PALM_PLANT_PROMPTS + NON_PLANT_PROMPTS
N1, N2, N3 = len(ARECANUT_PROMPTS), len(NON_PALM_PLANT_PROMPTS), len(NON_PLANT_PROMPTS)

# Decision thresholds
REJECT_NON_PLANT_THRESHOLD = 0.55
REJECT_NON_PALM_THRESHOLD  = 0.50
ACCEPT_ARECANUT_THRESHOLD   = 0.30
MIN_IMAGE_DIM = 50

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_clip():
    from transformers import TFCLIPModel, CLIPProcessor
    print("  Loading CLIP ViT-B/32...")
    model = TFCLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    print("  CLIP loaded.")
    return model, processor

def run_clip(model, processor, pil_img, prompts):
    inputs = processor(text=prompts, images=pil_img, return_tensors="tf", padding=True)
    outputs = model(**inputs)
    logits = outputs.logits_per_image.numpy()[0]
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)
    results = list(zip(prompts, probs))
    results.sort(key=lambda x: x[1], reverse=True)
    return results, probs

def make_decision(probs):
    p_arecanut = float(np.sum(probs[:N1]))
    p_non_palm = float(np.sum(probs[N1:N1+N2]))
    p_non_plant = float(np.sum(probs[N1+N2:]))
    if p_non_plant >= REJECT_NON_PLANT_THRESHOLD:
        return "REJECT", p_arecanut, p_non_palm, p_non_plant, "non-plant"
    if p_non_palm >= REJECT_NON_PALM_THRESHOLD:
        return "REJECT", p_arecanut, p_non_palm, p_non_plant, "non-palm-plant"
    if p_arecanut >= ACCEPT_ARECANUT_THRESHOLD and \
       p_arecanut > p_non_palm and p_arecanut > p_non_plant:
        return "ACCEPT", p_arecanut, p_non_palm, p_non_plant, ""
    return "REJECT", p_arecanut, p_non_palm, p_non_plant, "conservative"

def sample_arecanut_images(dataset_dir, n=3):
    results = []
    if not os.path.isdir(dataset_dir):
        return results
    for cls_dir in sorted(os.listdir(dataset_dir)):
        cls_path = os.path.join(dataset_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        imgs = sorted([f for f in os.listdir(cls_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for img_name in imgs[:n]:
            results.append((cls_dir, os.path.join(cls_path, img_name)))
    return results

def make_synthetic_images():
    images = []
    images.append(("syn_red", Image.new("RGB", (256, 256), (220, 30, 30))))
    images.append(("syn_green", Image.new("RGB", (256, 256), (30, 200, 30))))
    images.append(("syn_blue", Image.new("RGB", (256, 256), (30, 30, 220))))
    images.append(("syn_white", Image.new("RGB", (256, 256), (255, 255, 255))))
    images.append(("syn_noise", Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))))
    images.append(("syn_skin", Image.new("RGB", (256, 256), (210, 170, 140))))
    images.append(("syn_sky", Image.new("RGB", (256, 256), (135, 206, 235))))
    images.append(("syn_wood", Image.new("RGB", (256, 256), (139, 69, 19))))
    return images

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  ArecaNut AI - CLIP Zero-Shot Validation Test v2")
    print("  Prompts: 8 arecanut + 6 non-palm-plant + 12 non-plant = 26 total")
    print("  Key change: banana/coconut trees REMOVED from reject prompts")
    print("=" * 78)

    try:
        model, processor = load_clip()
    except Exception as exc:
        print(f"\n  [FATAL] CLIP load failed: {exc}")
        traceback.print_exc()
        return

    arecanut_images = sample_arecanut_images(DATASET_TEST_DIR, n=3)
    synthetic_images = make_synthetic_images()
    print(f"\n  Arecanut images: {len(arecanut_images)} ({len(arecanut_images)//3} classes x 3)")
    print(f"  Synthetic controls: {len(synthetic_images)}")

    all_results = []

    # ── Arecanut images ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  ARECANUT IMAGES (expected: all ACCEPT)")
    print("=" * 78)

    for class_name, img_path in arecanut_images:
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
                print(f"  [SKIP] {class_name}: image too small ({w}x{h})")
                all_results.append(("arecanut", class_name, "SKIP", 0, 0, 0, []))
                continue

            results, probs = run_clip(model, processor, img, ALL_PROMPTS)
            decision, pa, pn, pp, reason = make_decision(probs)

            top5 = ", ".join(f"{p[0]}({p[1]:.1%})" for p in results[:5])
            status = "PASS" if decision == "ACCEPT" else "FALSE_REJECT"
            print(f"  [{status:12s}] {class_name} ({w}x{h})")
            print(f"    P(arecanut)={pa:.1%} P(non_palm)={pn:.1%} P(non_plant)={pp:.1%}")
            print(f"    Top-5: {top5}")
            if reason:
                print(f"    Reject reason: {reason}")
            print()
            all_results.append(("arecanut", class_name, decision, pa, pn, pp, results[:5]))
        except Exception as e:
            print(f"  [ERROR] {class_name}: {e}\n")
            all_results.append(("arecanut", class_name, "ERROR", 0, 0, 0, []))

    # ── Synthetic controls ──────────────────────────────────────────────────
    print("=" * 78)
    print("  SYNTHETIC CONTROLS (expected: all REJECT)")
    print("=" * 78)

    for label, img in synthetic_images:
        try:
            results, probs = run_clip(model, processor, img, ALL_PROMPTS)
            decision, pa, pn, pp, reason = make_decision(probs)
            top5 = ", ".join(f"{p[0]}({p[1]:.1%})" for p in results[:5])
            status = "PASS" if decision == "REJECT" else "FALSE_ACCEPT"
            print(f"  [{status:12s}] {label}")
            print(f"    P(arecanut)={pa:.1%} P(non_palm)={pn:.1%} P(non_plant)={pp:.1%}")
            print(f"    Top-5: {top5}")
            if reason:
                print(f"    Reject reason: {reason}")
            print()
            all_results.append(("synthetic", label, decision, pa, pn, pp, results[:5]))
        except Exception as e:
            print(f"  [ERROR] {label}: {e}\n")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)

    arec_results = [r for r in all_results if r[0] == "arecanut"]
    syn_results  = [r for r in all_results if r[0] == "synthetic"]

    arec_accept = sum(1 for r in arec_results if r[2] == "ACCEPT")
    arec_reject = sum(1 for r in arec_results if r[2] == "REJECT")
    arec_skip   = sum(1 for r in arec_results if r[2] == "SKIP")
    syn_reject  = sum(1 for r in syn_results if r[2] == "REJECT")
    syn_accept  = sum(1 for r in syn_results if r[2] == "ACCEPT")

    tested_arec = arec_accept + arec_reject
    print(f"\n  Arecanut images: {tested_arec} tested, {arec_skip} skipped (too small)")
    print(f"    Accepted (correct):      {arec_accept}")
    print(f"    Rejected (false reject): {arec_reject}")
    if tested_arec:
        print(f"    Acceptance rate: {arec_accept/tested_arec*100:.0f}%")

    print(f"\n  Synthetic controls: {len(syn_results)} tested")
    print(f"    Rejected (correct):     {syn_reject}")
    print(f"    Accepted (false accept): {syn_accept}")
    if syn_results:
        print(f"    Rejection rate: {syn_reject/len(syn_results)*100:.0f}%")

    # Per-class breakdown
    print(f"\n  Per-class arecanut results:")
    class_stats = {}
    for cat, label, decision, pa, pn, pp, top5 in arec_results:
        if label not in class_stats:
            class_stats[label] = {"accept": 0, "reject": 0, "skip": 0}
        if decision == "ACCEPT":
            class_stats[label]["accept"] += 1
        elif decision == "REJECT":
            class_stats[label]["reject"] += 1
        else:
            class_stats[label]["skip"] += 1

    for cls in sorted(class_stats.keys()):
        s = class_stats[cls]
        total = s["accept"] + s["reject"]
        if total > 0:
            print(f"    {cls:<25}  accept={s['accept']} reject={s['reject']} ({s['accept']}/{total})")
        else:
            print(f"    {cls:<25}  skipped (too small)")

    # False rejections
    false_rejects = [r for r in arec_results if r[2] == "REJECT"]
    if false_rejects:
        print(f"\n  FALSE REJECTIONS:")
        for cat, label, decision, pa, pn, pp, top5 in false_rejects:
            print(f"    {label}: P(arecanut)={pa:.1%} top={top5[0][0]}({top5[0][1]:.1%})")

    # False acceptances
    false_accepts = [r for r in syn_results if r[2] == "ACCEPT"]
    if false_accepts:
        print(f"\n  FALSE ACCEPTANCES (synthetic):")
        for cat, label, decision, pa, pn, pp, top5 in false_accepts:
            print(f"    {label}: P(arecanut)={pa:.1%} top={top5[0][0]}({top5[0][1]:.1%})")

    print(f"\n  {'=' * 40}")
    if arec_reject == 0 and arec_skip == 0:
        print(f"  [OK] All {arec_accept} arecanut images accepted.")
    elif arec_reject > 0:
        print(f"  [FAIL] {arec_reject} arecanut image(s) falsely rejected.")
    if syn_accept == 0:
        print(f"  [OK] All {syn_reject} synthetic controls rejected.")
    else:
        print(f"  [FAIL] {syn_accept} synthetic image(s) falsely accepted.")

    print(f"\n  LIMITATION: CLIP cannot distinguish arecanut from other palm")
    print(f"  species (coconut, banana, date palm). Banana and coconut prompts")
    print(f"  were REMOVED from the reject list to prevent false rejections of")
    print(f"  genuine arecanut images. This means banana/coconut tree photos")
    print(f"  would be ACCEPTED by this guard.")
    print(f"\n  Real unrelated images (people, animals, vehicles) were NOT")
    print(f"  tested (network blocked downloads). Synthetic controls only.")
    print("=" * 78)

if __name__ == "__main__":
    main()
