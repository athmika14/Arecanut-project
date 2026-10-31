# -*- coding: utf-8 -*-
"""
test_clip_guard.py — ArecaNut AI: CLIP Zero-Shot Validation Test
=================================================================
Standalone test script.  Does NOT modify any project files.

Tests whether OpenAI CLIP (ViT-B/32) can distinguish arecanut images
from unrelated images using zero-shot text-prompt classification.

Test images:
  1. Real arecanut images: 2 per class x 9 classes = 18 images
     (sampled from the existing disease dataset)
  2. Synthetic controls: solid colors, noise, gradient = 5 images
  3. Downloaded unrelated images (if network available): people,
     animals, buildings, vehicles, food, other plants

Scoring:
  25 candidate text prompts in 3 tiers:
    Tier 1 (arecanut-positive):  7 prompts
    Tier 2 (other-plant reject): 8 prompts
    Tier 3 (non-plant reject):  10 prompts

  P(arecanut)   = sum of Tier 1 softmax probabilities
  P(other_plant) = sum of Tier 2 softmax probabilities
  P(non_plant)  = sum of Tier 3 softmax probabilities

  Decision:
    ACCEPT  if P(arecanut) is the highest group AND P(arecanut) >= 0.30
    REJECT  if P(non_plant) >= 0.50 or P(other_plant) >= 0.50
    REJECT  (conservative) if no clear winner

Usage:
    python ml/validation/training/test_clip_guard.py
"""

import os
import sys
import io
import warnings
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.dirname(SCRIPT_DIR)
ML_DIR         = os.path.dirname(VALIDATION_DIR)
PROJECT_DIR    = os.path.dirname(ML_DIR)

DATASET_TEST_DIR = os.path.join(
    ML_DIR, "disease", "dataset", "Arecanut_dataset",
    "Arecanut_dataset", "test"
)

# ── CLIP model name ──────────────────────────────────────────────────────────

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ── Candidate text prompts (25 total) ─────────────────────────────────────────

# Tier 1: arecanut-positive (7 prompts)
ARECANUT_PROMPTS = [
    "a photo of an arecanut palm tree",
    "a photo of areca nuts on a palm tree",
    "a photo of a palm tree trunk",
    "a photo of a palm leaf",
    "a photo of a diseased palm leaf",
    "a photo of a tropical palm plantation",
    "a photo of a palm tree stem",
]

# Tier 2: other-plant reject (8 prompts)
OTHER_PLANT_PROMPTS = [
    "a photo of a rose flower",
    "a photo of a tomato plant",
    "a photo of a banana tree",
    "a photo of a wheat field",
    "a photo of a cactus",
    "a photo of a sunflower",
    "a photo of a coconut tree",
    "a photo of a potted houseplant",
]

# Tier 3: non-plant reject (10 prompts)
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
]

ALL_PROMPTS = ARECANUT_PROMPTS + OTHER_PLANT_PROMPTS + NON_PLANT_PROMPTS
N_TIER1 = len(ARECANUT_PROMPTS)   # 7
N_TIER2 = len(OTHER_PLANT_PROMPTS) # 8
N_TIER3 = len(NON_PLANT_PROMPTS)   # 10

# Decision thresholds
ACCEPT_THRESHOLD = 0.30   # P(arecanut) must be >= this to accept
REJECT_THRESHOLD = 0.50   # P(non_plant) or P(other_plant) >= this to reject

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_clip_model():
    """Load CLIP ViT-B/32 via HuggingFace transformers (TF backend)."""
    from transformers import TFCLIPModel, CLIPProcessor

    print("  Loading CLIP model (first run downloads ~338 MB)...")
    model = TFCLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    print("  CLIP model loaded successfully.")
    return model, processor


def run_clip(model, processor, pil_image, prompts):
    """
    Run CLIP zero-shot classification on a single PIL image.
    Returns: list of (prompt, probability) sorted by probability descending.
    """
    # CLIPProcessor expects PIL images or tensors
    inputs = processor(
        text=prompts,
        images=pil_image,
        return_tensors="tf",
        padding=True,
    )

    outputs = model(**inputs)
    # logits_per_image shape: (1, num_prompts)
    logits = outputs.logits_per_image.numpy()[0]

    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # Sort by probability descending
    results = list(zip(prompts, probs))
    results.sort(key=lambda x: x[1], reverse=True)
    return results, probs


def make_decision(probs):
    """
    Given the raw probability array (length 25), compute group probabilities
    and return (decision, p_arecanut, p_other_plant, p_non_plant).

    decision: "ACCEPT" or "REJECT"
    """
    p_arecanut    = float(np.sum(probs[:N_TIER1]))
    p_other_plant = float(np.sum(probs[N_TIER1:N_TIER1 + N_TIER2]))
    p_non_plant   = float(np.sum(probs[N_TIER1 + N_TIER2:]))

    # Decision logic
    if p_non_plant >= REJECT_THRESHOLD:
        return "REJECT", p_arecanut, p_other_plant, p_non_plant
    if p_other_plant >= REJECT_THRESHOLD:
        return "REJECT", p_arecanut, p_other_plant, p_non_plant
    if p_arecanut >= ACCEPT_THRESHOLD and \
       p_arecanut > p_other_plant and p_arecanut > p_non_plant:
        return "ACCEPT", p_arecanut, p_other_plant, p_non_plant
    # Conservative: reject if no clear winner
    return "REJECT", p_arecanut, p_other_plant, p_non_plant


def sample_arecanut_images(dataset_dir, n_per_class=2):
    """
    Sample n_per_class images from each of the 9 arecanut classes.
    Returns list of (class_name, image_path).
    """
    results = []
    if not os.path.isdir(dataset_dir):
        print(f"  [WARNING] Dataset dir not found: {dataset_dir}")
        return results

    for cls_dir in sorted(os.listdir(dataset_dir)):
        cls_path = os.path.join(dataset_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        imgs = sorted([f for f in os.listdir(cls_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for img_name in imgs[:n_per_class]:
            results.append((cls_dir, os.path.join(cls_path, img_name)))
    return results


def make_synthetic_images():
    """
    Create synthetic control images with Pillow.
    Returns list of (label, PIL.Image).
    """
    images = []

    # Solid red
    images.append(("synthetic_red", Image.new("RGB", (256, 256), (220, 30, 30))))
    # Solid green
    images.append(("synthetic_green", Image.new("RGB", (256, 256), (30, 200, 30))))
    # Solid blue
    images.append(("synthetic_blue", Image.new("RGB", (256, 256), (30, 30, 220))))
    # Solid white
    images.append(("synthetic_white", Image.new("RGB", (256, 256), (255, 255, 255))))
    # Random noise
    noise_arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    images.append(("synthetic_noise", Image.fromarray(noise_arr)))

    return images


def try_download_images():
    """
    Try to download a few small unrelated images from public URLs.
    Returns list of (label, PIL.Image) or empty list if network fails.
    """
    images = []
    urls = [
        # Try Wikimedia Commons thumbnails (small, public domain)
        ("person",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/STEVE_JOBS_1.jpg/200px-STEVE_JOBS_1.jpg"),
        ("dog",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/200px-YellowLabradorLooking_new.jpg"),
        ("building",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Bodleian-library-oxford.jpg/200px-Bodleian-library-oxford.jpg"),
        ("car",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/VW_Golf_1.jpg/200px-VW_Golf_1.jpg"),
        ("rose",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Rosa_rubiginosa_1.jpg/200px-Rosa_rubiginosa_1.jpg"),
        ("banana",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/200px-Banana-Single.jpg"),
        ("food_pizza",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Supreme_pizza.jpg/200px-Supreme_pizza.jpg"),
        ("landscape",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Matterhorn_from_Dom.jpg/200px-Matterhorn_from_Dom.jpg"),
        ("cat",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/200px-Cat_November_2010-1a.jpg"),
        ("phone",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/IPhone_4_no_bg.png/200px-IPhone_4_no_bg.png"),
    ]

    import urllib.request
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for label, url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=10, context=ctx)
            img_data = resp.read()
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append((label, img))
        except Exception as e:
            print(f"    [SKIP] Could not download '{label}': {type(e).__name__}")
            continue

    return images


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  ArecaNut AI — CLIP Zero-Shot Validation Test")
    print("=" * 78)
    print()

    # ── Step 1: Load CLIP model ──────────────────────────────────────────────
    print("── Step 1: Loading CLIP model ──")
    try:
        model, processor = load_clip_model()
    except Exception as exc:
        print(f"\n  [FATAL] Could not load CLIP model: {exc}")
        traceback.print_exc()
        return

    # ── Step 2: Gather test images ──────────────────────────────────────────
    print("\n── Step 2: Gathering test images ──")

    # 2a: Real arecanut images (2 per class)
    arecanut_images = sample_arecanut_images(DATASET_TEST_DIR, n_per_class=2)
    print(f"  Real arecanut images: {len(arecanut_images)} ({len(arecanut_images)//2} classes x 2)")

    # 2b: Synthetic control images
    synthetic_images = make_synthetic_images()
    print(f"  Synthetic control images: {len(synthetic_images)}")

    # 2c: Try downloading real unrelated images
    print(f"  Attempting to download unrelated test images...")
    downloaded_images = try_download_images()
    print(f"  Downloaded unrelated images: {len(downloaded_images)}")

    # ── Step 3: Run CLIP on all images ──────────────────────────────────────
    print("\n── Step 3: Running CLIP zero-shot classification ──\n")

    all_results = []  # (category, image_label, decision, p_arec, p_other, p_non_plant, top5)

    # 3a: Arecanut images (should all be ACCEPTED)
    print("=" * 78)
    print("  CATEGORY: ARECANUT IMAGES (expected: all ACCEPT)")
    print("=" * 78)

    for class_name, img_path in arecanut_images:
        try:
            img = Image.open(img_path).convert("RGB")
            results, probs = run_clip(model, processor, img, ALL_PROMPTS)
            decision, p_a, p_o, p_n = make_decision(probs)

            top5_str = ", ".join(f"{p[0]}({p[1]:.1%})" for p in results[:5])
            status = "PASS" if decision == "ACCEPT" else "FALSE_REJECT"
            print(f"  [{status:12s}] {class_name}")
            print(f"    P(arecanut)={p_a:.1%}  P(other_plant)={p_o:.1%}  P(non_plant)={p_n:.1%}")
            print(f"    Top-5: {top5_str}")
            print()

            all_results.append(("arecanut", class_name, decision, p_a, p_o, p_n, results[:5]))
        except Exception as e:
            print(f"  [ERROR] {class_name}: {e}")
            print()

    # 3b: Synthetic control images (should be REJECTED)
    print("=" * 78)
    print("  CATEGORY: SYNTHETIC CONTROLS (expected: all REJECT)")
    print("=" * 78)

    for label, img in synthetic_images:
        try:
            results, probs = run_clip(model, processor, img, ALL_PROMPTS)
            decision, p_a, p_o, p_n = make_decision(probs)

            top5_str = ", ".join(f"{p[0]}({p[1]:.1%})" for p in results[:5])
            status = "PASS" if decision == "REJECT" else "FALSE_ACCEPT"
            print(f"  [{status:12s}] {label}")
            print(f"    P(arecanut)={p_a:.1%}  P(other_plant)={p_o:.1%}  P(non_plant)={p_n:.1%}")
            print(f"    Top-5: {top5_str}")
            print()

            all_results.append(("synthetic", label, decision, p_a, p_o, p_n, results[:5]))
        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
            print()

    # 3c: Downloaded unrelated images (should be REJECTED)
    if downloaded_images:
        print("=" * 78)
        print("  CATEGORY: REAL UNRELATED IMAGES (expected: all REJECT)")
        print("=" * 78)

        for label, img in downloaded_images:
            try:
                results, probs = run_clip(model, processor, img, ALL_PROMPTS)
                decision, p_a, p_o, p_n = make_decision(probs)

                top5_str = ", ".join(f"{p[0]}({p[1]:.1%})" for p in results[:5])
                status = "PASS" if decision == "REJECT" else "FALSE_ACCEPT"
                print(f"  [{status:12s}] {label}")
                print(f"    P(arecanut)={p_a:.1%}  P(other_plant)={p_o:.1%}  P(non_plant)={p_n:.1%}")
                print(f"    Top-5: {top5_str}")
                print()

                all_results.append(("unrelated", label, decision, p_a, p_o, p_n, results[:5]))
            except Exception as e:
                print(f"  [ERROR] {label}: {e}")
                print()
    else:
        print("\n  [NOTE] No downloaded unrelated images. Only synthetic controls tested.")
        print("         Real-image rejection performance is NOT verified.")
        print()

    # ── Step 4: Summary ─────────────────────────────────────────────────────
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)

    arecanut_results = [r for r in all_results if r[0] == "arecanut"]
    synthetic_results = [r for r in all_results if r[0] == "synthetic"]
    unrelated_results = [r for r in all_results if r[0] == "unrelated"]

    arecanut_accept = sum(1 for r in arecanut_results if r[2] == "ACCEPT")
    arecanut_reject = sum(1 for r in arecanut_results if r[2] == "REJECT")
    synthetic_reject = sum(1 for r in synthetic_results if r[2] == "REJECT")
    synthetic_accept = sum(1 for r in synthetic_results if r[2] == "ACCEPT")
    unrelated_reject = sum(1 for r in unrelated_results if r[2] == "REJECT")
    unrelated_accept = sum(1 for r in unrelated_results if r[2] == "ACCEPT")

    print(f"\n  Arecanut images ({len(arecanut_results)} tested):")
    print(f"    Accepted (correct):    {arecanut_accept}")
    print(f"    Rejected (false reject): {arecanut_reject}")
    if arecanut_results:
        print(f"    Acceptance rate: {arecanut_accept/len(arecanut_results)*100:.0f}%")

    print(f"\n  Synthetic controls ({len(synthetic_results)} tested):")
    print(f"    Rejected (correct):    {synthetic_reject}")
    print(f"    Accepted (false accept): {synthetic_accept}")
    if synthetic_results:
        print(f"    Rejection rate: {synthetic_reject/len(synthetic_results)*100:.0f}%")

    print(f"\n  Real unrelated images ({len(unrelated_results)} tested):")
    print(f"    Rejected (correct):    {unrelated_reject}")
    print(f"    Accepted (false accept): {unrelated_accept}")
    if unrelated_results:
        print(f"    Rejection rate: {unrelated_reject/len(unrelated_results)*100:.0f}%")

    # Per-class arecanut breakdown
    print(f"\n  Per-class arecanut results:")
    class_stats = {}
    for cat, label, decision, p_a, p_o, p_n, top5 in arecanut_results:
        if label not in class_stats:
            class_stats[label] = {"accept": 0, "reject": 0}
        if decision == "ACCEPT":
            class_stats[label]["accept"] += 1
        else:
            class_stats[label]["reject"] += 1

    for cls in sorted(class_stats.keys()):
        s = class_stats[cls]
        total = s["accept"] + s["reject"]
        print(f"    {cls:<25}  accept={s['accept']} reject={s['reject']} "
              f"({s['accept']}/{total})")

    # False rejections detail
    false_rejects = [r for r in arecanut_results if r[2] == "REJECT"]
    if false_rejects:
        print(f"\n  FALSE REJECTIONS (arecanut images wrongly rejected):")
        for cat, label, decision, p_a, p_o, p_n, top5 in false_rejects:
            print(f"    {label}: P(arecanut)={p_a:.1%} top={top5[0][0]}({top5[0][1]:.1%})")

    # False acceptances detail
    false_accepts = [r for r in synthetic_results + unrelated_results if r[2] == "ACCEPT"]
    if false_accepts:
        print(f"\n  FALSE ACCEPTANCES (unrelated images wrongly accepted):")
        for cat, label, decision, p_a, p_o, p_n, top5 in false_accepts:
            print(f"    {label}: P(arecanut)={p_a:.1%} top={top5[0][0]}({top5[0][1]:.1%})")

    # Final verdict
    print(f"\n  {'=' * 40}")
    all_arecanut_pass = arecanut_reject == 0
    all_unrelated_pass = (len(unrelated_results) == 0 or unrelated_accept == 0)
    all_synthetic_pass = synthetic_accept == 0

    if all_arecanut_pass:
        print(f"  [OK] All {arecanut_accept} arecanut images accepted.")
    else:
        print(f"  [FAIL] {arecanut_reject} arecanut image(s) falsely rejected.")

    if all_synthetic_pass:
        print(f"  [OK] All {synthetic_reject} synthetic controls rejected.")
    else:
        print(f"  [FAIL] {synthetic_accept} synthetic image(s) falsely accepted.")

    if unrelated_results:
        if all_unrelated_pass:
            print(f"  [OK] All {unrelated_reject} real unrelated images rejected.")
        else:
            print(f"  [FAIL] {unrelated_accept} real unrelated image(s) falsely accepted.")
    else:
        print(f"  [UNTESTED] No real unrelated images available for testing.")

    print(f"\n  NOTE: Other plant species (rose, banana, coconut) rejection")
    print(f"  was only tested if downloaded images were available.")
    print(f"  CLIP cannot distinguish arecanut from other palm species.")
    print("=" * 78)


if __name__ == "__main__":
    main()
