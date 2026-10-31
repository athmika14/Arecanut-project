"""Quick inference smoke-test — no Flask, just raw model + Pillow."""
import os, sys, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mobilenetv2_best.keras")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "class_indices.json")

print("Loading model ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("  Input :", model.input_shape)
print("  Output:", model.output_shape)

with open(INDEX_PATH, encoding="utf-8") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}
print("  Classes:", idx_to_class)

# Build a synthetic 224x224 RGB image and run inference
img = Image.new("RGB", (256, 256), color=(120, 180, 60))   # greenish leaf-ish
img = img.resize((224, 224), Image.BILINEAR)
arr = np.array(img, dtype=np.float32)
arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)   # scale to [-1,1]
arr = np.expand_dims(arr, axis=0)                                  # (1,224,224,3)

preds = model.predict(arr, verbose=0)
idx   = int(np.argmax(preds[0]))
conf  = float(preds[0][idx])
label = idx_to_class[idx]

print(f"\nPrediction : {label}  (index={idx})")
print(f"Confidence : {conf:.4f}  ({conf*100:.2f}%)")
print("All scores :")
for i, score in enumerate(preds[0]):
    print(f"  {i:>2}  {idx_to_class[i]:<25}  {score:.4f}")

print("\nSMOKE TEST PASSED")
