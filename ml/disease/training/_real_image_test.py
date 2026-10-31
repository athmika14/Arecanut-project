# -*- coding: utf-8 -*-
"""
_real_image_test.py  —  ArecaNut AI: Real Image Prediction Audit
=================================================================
READ-ONLY.  Does not modify any application source or dataset file.

Tests prediction accuracy using real labelled images from the
test split manifest (manifest_test.csv).

Strategy:
  - Read image paths and true labels from manifest_test.csv
  - Sample 3 images per class (27 images max) to keep runtime short
  - Run each through the real Flask API via test client
  - Compare predicted_class to the true label from the manifest
  - Report per-class results and overall accuracy on the sample
  - Report whether disease vs healthy classification was correct for each

This is NOT a complete evaluation (that is already recorded in
evaluation_metrics.json with 1,943 test images). It is a spot-check
to verify the inference pipeline works on real labelled images.
"""
import os, sys, csv, json, time, io
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings; warnings.filterwarnings("ignore")

PROJECT_DIR = r"d:\mca project\ArecaNut_AI"
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

MANIFEST = os.path.join(PROJECT_DIR, "ml", "disease", "prepared_data", "manifest_test.csv")
SAMPLES_PER_CLASS = 3   # keep the test fast; enough to verify routing

DISEASE_CLASSES = {"Mahali_Koleroga","Stem_bleeding","bud borer","stem cracking","yellow leaf disease"}
HEALTHY_CLASSES = {"Healthy_Nut","Healthy_Leaf","Healthy_Trunk","healthy_foot"}

# ── Load model and create test client ─────────────────────────────────────────
from app import create_app
app = create_app()
from routes.disease import _model
assert _model is not None, "Model failed to load"

# ── Sample images from manifest ───────────────────────────────────────────────
# Collect up to SAMPLES_PER_CLASS real test images per class
by_class = {}
with open(MANIFEST, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cls = row["class_label"]
        path = row["image_path"]
        if not os.path.isfile(path):
            continue
        if cls not in by_class:
            by_class[cls] = []
        if len(by_class[cls]) < SAMPLES_PER_CLASS:
            by_class[cls].append(path)

print(f"Classes with images found: {sorted(by_class.keys())}")
total_samples = sum(len(v) for v in by_class.values())
print(f"Total test images sampled: {total_samples}")
print()

# ── Run predictions ────────────────────────────────────────────────────────────
results = []
with app.test_client() as client:
    for true_class, paths in sorted(by_class.items()):
        for img_path in paths:
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            fname = os.path.basename(img_path)
            ext = fname.rsplit(".", 1)[-1].lower()
            mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(ext,"image/jpeg")

            data = {"image": (io.BytesIO(img_bytes), fname, mime)}
            resp = client.post("/api/disease/predict", data=data, content_type="multipart/form-data")
            body = resp.get_json()

            if resp.status_code == 200:
                pred = body["predicted_class"]
                conf = body["confidence"]
                correct = (pred == true_class)

                true_status = "disease" if true_class in DISEASE_CLASSES else ("healthy" if true_class in HEALTHY_CLASSES else "unknown")
                pred_status = "disease" if pred in DISEASE_CLASSES else ("healthy" if pred in HEALTHY_CLASSES else "unknown")
                status_correct = (true_status == pred_status)

                results.append({
                    "true_class": true_class,
                    "pred_class": pred,
                    "confidence": conf,
                    "class_correct": correct,
                    "true_status": true_status,
                    "pred_status": pred_status,
                    "status_correct": status_correct,
                    "http_status": 200,
                    "filename": fname[:60],
                })
                marker = "PASS" if correct else "FAIL"
                print(f"  {marker} [{true_class}] -> {pred}  ({conf*100:.1f}%)  status:{pred_status}")
            else:
                results.append({
                    "true_class": true_class, "pred_class": "ERROR",
                    "confidence": 0, "class_correct": False,
                    "true_status": "unknown", "pred_status": "ERROR",
                    "status_correct": False,
                    "http_status": resp.status_code,
                    "filename": fname[:60],
                })
                print(f"  [ERROR] [{true_class}] HTTP {resp.status_code}: {body}")

# ── Summarise ─────────────────────────────────────────────────────────────────
class_correct = sum(1 for r in results if r["class_correct"])
status_correct = sum(1 for r in results if r["status_correct"])
total = len(results)

print()
print(f"Class-level accuracy on {total} real test images: {class_correct}/{total} ({class_correct/total*100:.1f}%)")
print(f"Disease/healthy status accuracy:                  {status_correct}/{total} ({status_correct/total*100:.1f}%)")

# Per-class summary
print()
print("Per-class results:")
for cls in sorted(by_class.keys()):
    cls_results = [r for r in results if r["true_class"] == cls]
    ok = sum(1 for r in cls_results if r["class_correct"])
    print(f"  {cls:<25} {ok}/{len(cls_results)} correct")

# Save JSON for the report writer
out_path = os.path.join(PROJECT_DIR, "ml", "disease", "training", "_real_image_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"results": results, "class_correct": class_correct,
               "status_correct": status_correct, "total": total}, f, indent=2)
print(f"\nResults saved to: {out_path}")
