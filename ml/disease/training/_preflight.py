# -*- coding: utf-8 -*-
"""Quick pre-flight check: CSV loading, path validation, model build."""
import os, sys, csv, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore")

PREPARED = r"d:\mca project\ArecaNut_AI\ml\disease\prepared_data"
EXPECTED = ["Healthy_Leaf","Healthy_Nut","Healthy_Trunk","Mahali_Koleroga",
            "Stem_bleeding","bud borer","healthy_foot","stem cracking",
            "yellow leaf disease"]

def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["image_path"], r["class_label"]))
    return rows

tr = load(os.path.join(PREPARED, "manifest_train.csv"))
va = load(os.path.join(PREPARED, "manifest_val.csv"))
te = load(os.path.join(PREPARED, "manifest_test.csv"))
print(f"CSV rows  train={len(tr)}  val={len(va)}  test={len(te)}"
      f"  total={len(tr)+len(va)+len(te)}")

missing = [p for p, _ in tr+va+te if not os.path.isfile(p)]
print(f"Missing paths : {len(missing)}")

bad = [l for _, l in tr+va+te if l not in EXPECTED]
print(f"Bad labels    : {len(bad)}")

found = sorted({l for _, l in tr})
print(f"Classes found : {found}")

if missing or bad:
    print("PRE-FLIGHT FAILED"); sys.exit(1)

import tensorflow as tf
print(f"TF version    : {tf.__version__}")

base = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3), include_top=False, weights="imagenet")
base.trainable = False
inp  = tf.keras.Input(shape=(224,224,3))
x    = base(inp, training=False)
x    = tf.keras.layers.GlobalAveragePooling2D()(x)
x    = tf.keras.layers.Dense(128, activation="relu")(x)
x    = tf.keras.layers.Dropout(0.3)(x)
out  = tf.keras.layers.Dense(9,  activation="softmax")(x)
model = tf.keras.Model(inp, out)
tp  = sum(int(tf.size(v)) for v in model.trainable_weights)
tot = sum(int(tf.size(v)) for v in model.weights)
print(f"Model built   : trainable={tp:,}  total={tot:,}")
print("PRE-FLIGHT PASSED — ready to train.")
