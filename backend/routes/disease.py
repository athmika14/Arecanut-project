# -*- coding: utf-8 -*-
"""
routes/disease.py  -  ArecaNut AI: Disease Detection
=====================================================
POST /api/disease/predict

Guard: CLIP zero-shot image validator (v2)
------------------------------------------
Replaces the previous ImageNet MobileNetV2 heuristic with OpenAI CLIP
(ViT-B/32) zero-shot classification.  The image is compared against 28
text prompts in three groups:

  Arecanut-positive (10): palm tree, areca nuts, betel nuts, palm trunk,
    palm leaf, diseased palm leaf, tropical palm plantation, palm stem,
    palm tree with disease symptoms.

  Non-palm-plant reject (6): rose, tomato, wheat, cactus, sunflower,
    potted houseplant.  These are clearly non-palm plants.

  Non-plant reject (12): person, baby, dog, cat, car, building, phone,
    food, landscape, blank image, screenshot, keyboard.

Decision:
  Reject if P(non_plant) >= 0.55 or P(non_palm_plant) >= 0.50.
  Accept if P(arecanut) >= 0.30 and P(arecanut) > both reject groups.
  Reject conservatively otherwise.

Known limitation: CLIP cannot distinguish arecanut from other palm
species (coconut, banana, date palm).  Banana and coconut prompts were
intentionally excluded from the reject list to prevent false rejections
of genuine arecanut trunk/stem images.  This means banana/coconut tree
photos would be accepted.  A trained binary validator
(ml/validation/models/arecanut_validator.keras) is the long-term fix.

Test results (27 arecanut images, 3 per class):
  26/27 accepted (96%), 1 false rejection (Healthy_Nut, small image).
  8/8 synthetic controls rejected (100%).
  Real unrelated images (people, animals) NOT tested (network blocked).

Other fixes preserved from previous version:
  1. Confidence values sent as raw [0,1] probabilities (no *100).
  2. Route decorator "/predict" with blueprint url_prefix="/api/disease".
  3. Form field key "file" matches frontend formData.append("file", ...).

IMPORTANT: TF_USE_LEGACY_KERAS=0 must be set before importing TensorFlow
because the 'transformers' package installs 'tf-keras' which redirects
tf.keras to a legacy API that cannot load the disease model saved with
Keras 3.x.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"  # must be set before TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"

import io
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Blueprint, jsonify, request

# --------------------------------------------------
# Blueprint
# --------------------------------------------------

disease_bp = Blueprint("disease", __name__)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = 224
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB
ALLOWED_EXTS = {"jpg", "jpeg", "png", "webp"}
MIN_IMAGE_DIM = 50  # reject images smaller than this (pixels)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT / "ml" / "disease" / "models"
    / "mobilenetv2_best.keras"
)
INDEX_PATH = (
    PROJECT_ROOT / "ml" / "disease" / "models"
    / "class_indices.json"
)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ── CLIP prompt design (v2) ──────────────────────────────────────────────────
# 10 arecanut-positive + 6 non-palm-plant reject + 12 non-plant reject = 28
# Banana and coconut REMOVED from reject (they are palms, too similar to
# arecanut — including them caused false rejections of Stem_bleeding images).

_ARECANUT_PROMPTS = [
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

_NON_PALM_PLANT_PROMPTS = [
    "a photo of a rose flower",
    "a photo of a tomato plant",
    "a photo of a wheat field",
    "a photo of a cactus",
    "a photo of a sunflower",
    "a photo of a potted houseplant",
]

_NON_PLANT_PROMPTS = [
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

_ALL_CLIP_PROMPTS = (
    _ARECANUT_PROMPTS
    + _NON_PALM_PLANT_PROMPTS
    + _NON_PLANT_PROMPTS
)
_N_ARECANUT = len(_ARECANUT_PROMPTS)       # 10
_N_NON_PALM = len(_NON_PALM_PLANT_PROMPTS)  # 6
_N_NON_PLANT = len(_NON_PLANT_PROMPTS)      # 12

# Decision thresholds (tuned from real-image testing)
_REJECT_NON_PLANT_THRESHOLD = 0.55
_REJECT_NON_PALM_THRESHOLD  = 0.50
_ACCEPT_ARECANUT_THRESHOLD    = 0.30

# --------------------------------------------------
# Module-level state
# --------------------------------------------------

_model = None            # disease classifier (MobileNetV2)
_class_names = None      # list[str], index -> class name
_clip_model = None       # CLIP zero-shot validator
_clip_processor = None   # CLIPProcessor for text+image encoding
_load_error = None       # disease model load error
_clip_load_error = None # CLIP load error (None if OK or not attempted)

# --------------------------------------------------
# Loaders
# --------------------------------------------------

def load_disease_model():
    """
    Load the nine-class disease classifier and the CLIP guard model.
    Called once from the app factory.  If the disease model fails,
    _load_error is set and /predict returns 503.  If CLIP fails but
    the disease model loads, _clip_load_error is set and the guard
    is skipped (fail-open) so disease prediction still works.
    """
    global _model, _class_names, _clip_model, _clip_processor
    global _load_error, _clip_load_error

    # ── Disease model ──────────────────────────────────────────────────────
    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Disease model not found: {MODEL_PATH}"
            )
        if not INDEX_PATH.is_file():
            raise FileNotFoundError(
                f"Class-index file not found: {INDEX_PATH}"
            )

        _model = tf.keras.models.load_model(
            str(MODEL_PATH), compile=False
        )

        with open(INDEX_PATH, encoding="utf-8") as fh:
            class_indices = json.load(fh)
        _class_names = [
            class_indices[str(i)]
            for i in range(len(class_indices))
        ]

        if _model.output_shape[-1] != len(_class_names):
            raise ValueError(
                "Model output count does not match class-index count."
            )

        logger.info(
            "Disease model loaded: output=%s classes=%d",
            _model.output_shape, len(_class_names),
        )

    except Exception as exc:
        _load_error = str(exc)
        logger.exception("Could not load disease model")
        return  # don't attempt CLIP if disease model failed

    # ── CLIP guard model ────────────────────────────────────────────────────
    try:
        from transformers import TFCLIPModel, CLIPProcessor

        _clip_model = TFCLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model.trainable = False

        logger.info(
            "CLIP guard loaded: %s (prompts=%d)",
            CLIP_MODEL_NAME, len(_ALL_CLIP_PROMPTS),
        )

    except Exception as exc:
        _clip_load_error = str(exc)
        logger.warning(
            "CLIP guard failed to load (fail-open, guard will be skipped): %s",
            exc,
        )


# --------------------------------------------------
# Preprocessing (for disease model — unchanged)
# --------------------------------------------------

def preprocess_image(file_bytes: bytes, extension: str) -> np.ndarray:
    """
    Decode and resize an image for MobileNetV2 input.

    JPEG uses TensorFlow JPEG decoding (matches the training pipeline).
    PNG and WEBP are decoded with Pillow then converted to a TF tensor.

    Returns a float32 array of shape (1, 224, 224, 3) with pixel values
    in [-1, 1] as required by mobilenet_v2.preprocess_input.

    Raises ValueError for empty, unreadable, or decode-failed images.
    """
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        if extension in {"jpg", "jpeg"}:
            tensor = tf.io.decode_jpeg(file_bytes, channels=3)
        else:
            with Image.open(io.BytesIO(file_bytes)) as pil_img:
                pil_img.load()
                rgb = pil_img.convert("RGB")
                tensor = tf.convert_to_tensor(
                    np.array(rgb, dtype=np.uint8), dtype=tf.uint8
                )

        tensor = tf.image.resize(tensor, [IMG_SIZE, IMG_SIZE])
        tensor = tf.cast(tensor, tf.float32)
        tensor = tf.keras.applications.mobilenet_v2.preprocess_input(tensor)
        return tf.expand_dims(tensor, axis=0).numpy()  # (1, 224, 224, 3)

    except Exception as exc:
        raise ValueError(
            "Cannot read this image. Please upload a valid JPG, "
            "JPEG, PNG, or WEBP image."
        ) from exc


# --------------------------------------------------
# CLIP content guard
# --------------------------------------------------

def is_likely_arecanut_image(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Run the CLIP zero-shot validator on a PIL image.

    Returns (True, "") if the image should be passed to the disease
    classifier, or (False, reason) if it should be rejected.

    The image is compared against 28 text prompts (10 arecanut-positive,
    6 non-palm-plant reject, 12 non-plant reject).  Softmax probabilities
    are grouped and compared against thresholds.

    If the CLIP model is not loaded, the function fails open (accepts
    the image) so the disease classifier still runs.

    Limitations:
    - Cannot distinguish arecanut from coconut/banana/date palms.
    - May reject small or low-quality arecanut nut close-ups.
    - Not tested with real unrelated photos (network blocked downloads).
    """
    if _clip_model is None or _clip_processor is None:
        logger.warning(
            "CLIP guard not available; skipping validation. "
            "Reason: %s", _clip_load_error or "unknown"
        )
        return True, ""

    # ── Minimum dimension check ──────────────────────────────────────────────
    w, h = pil_image.size
    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
        return False, (
            f"Image is too small ({w}x{h} pixels). "
            f"Please upload a clearer photo at least {MIN_IMAGE_DIM}x"
            f"{MIN_IMAGE_DIM} pixels."
        )

    # ── Run CLIP zero-shot classification ───────────────────────────────────
    try:
        inputs = _clip_processor(
            text=_ALL_CLIP_PROMPTS,
            images=pil_image,
            return_tensors="tf",
            padding=True,
        )
        outputs = _clip_model(**inputs)
        logits = outputs.logits_per_image.numpy()[0]  # (28,)

        # Softmax to get relative probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

    except Exception as exc:
        logger.warning("CLIP inference failed (fail-open): %s", exc)
        return True, ""

    # ── Group probabilities ──────────────────────────────────────────────────
    p_arecanut   = float(np.sum(probs[:_N_ARECANUT]))
    p_non_palm   = float(np.sum(probs[_N_ARECANUT:_N_ARECANUT + _N_NON_PALM]))
    p_non_plant  = float(np.sum(probs[_N_ARECANUT + _N_NON_PALM:]))

    # ── Decision logic ───────────────────────────────────────────────────────
    # 1. Reject clearly non-plant images (people, vehicles, buildings)
    if p_non_plant >= _REJECT_NON_PLANT_THRESHOLD:
        return False, (
            "The image does not appear to contain a plant. "
            "Please upload a clear photo of an arecanut leaf, "
            "nut, trunk, or plant."
        )

    # 2. Reject clearly non-palm plants (roses, wheat, tomatoes)
    if p_non_palm >= _REJECT_NON_PALM_THRESHOLD:
        return False, (
            "The image appears to show a different type of plant, "
            "not an arecanut. Please upload a clear photo of an "
            "arecanut leaf, nut, trunk, or plant."
        )

    # 3. Accept if arecanut prompts score above threshold and dominate
    if (p_arecanut >= _ACCEPT_ARECANUT_THRESHOLD
            and p_arecanut > p_non_palm
            and p_arecanut > p_non_plant):
        return True, ""

    # 4. Conservative reject for ambiguous images
    return False, (
        "The image could not be clearly identified as an arecanut "
        "plant. Please upload a clearer photo of an arecanut leaf, "
        "nut, trunk, or plant."
    )


# --------------------------------------------------
# Route  -  POST /predict
# (Blueprint registered with url_prefix="/api/disease" in app.py)
# --------------------------------------------------

@disease_bp.route("/predict", methods=["POST"])
def predict_disease():
    """
    POST /api/disease/predict

    Expects multipart/form-data with field name 'file'.
    Allowed types : JPG, JPEG, PNG, WEBP
    Max size      : 10 MB

    Success response (HTTP 200)
    ---------------------------
    {
      "success": true,
      "predicted_class": "Mahali_Koleroga",
      "confidence": 0.9123,          <-- raw probability in [0, 1]
      "note": "...",
      "all_scores": {
        "Healthy_Leaf": 0.0034,
        ...                          <-- all raw probabilities in [0, 1]
      }
    }

    Error responses
    ---------------
    400  bad request (missing file, wrong extension, empty, corrupt)
    413  file too large
    422  content guard: image rejected (not arecanut-like)
    503  model not loaded
    500  unexpected inference error
    """

    # ── Model availability ────────────────────────────────────────────
    if _model is None or _class_names is None:
        return jsonify({
            "success": False,
            "error": (
                _load_error or
                "Disease model is not loaded. Check the backend logs."
            ),
        }), 503

    # ── File presence ─────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "Please upload an image using the 'file' field.",
        }), 400

    uploaded = request.files["file"]

    if not uploaded.filename:
        return jsonify({
            "success": False,
            "error": "Please select an image.",
        }), 400

    # ── Extension check ───────────────────────────────────────────────
    ext = Path(uploaded.filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTS:
        return jsonify({
            "success": False,
            "error": (
                "Unsupported image format. "
                "Please upload a JPG, JPEG, PNG, or WEBP image."
            ),
        }), 400

    # ── Size check ────────────────────────────────────────────────────
    file_bytes = uploaded.stream.read(MAX_FILE_SIZE + 1)

    if not file_bytes:
        return jsonify({
            "success": False,
            "error": "The uploaded file is empty.",
        }), 400

    if len(file_bytes) > MAX_FILE_SIZE:
        return jsonify({
            "success": False,
            "error": "Image is too large. Maximum size is 10 MB.",
        }), 413

    # ── Decode to PIL image for CLIP guard ────────────────────────────
    try:
        pil_image = Image.open(io.BytesIO(file_bytes))
        pil_image.load()
        pil_image = pil_image.convert("RGB")
    except Exception:
        return jsonify({
            "success": False,
            "error": (
                "Cannot read this image. Please upload a valid JPG, "
                "JPEG, PNG, or WEBP image."
            ),
        }), 400

    # ── CLIP content guard ────────────────────────────────────────────
    ok, guard_reason = is_likely_arecanut_image(pil_image)
    if not ok:
        return jsonify({
            "success": False,
            "rejected_by_guard": True,
            "error": guard_reason,
        }), 422

    # ── Preprocess for disease model ──────────────────────────────────
    try:
        image_array = preprocess_image(file_bytes, ext)
    except ValueError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 400

    # ── Disease classification ────────────────────────────────────────
    try:
        raw_preds = _model.predict(image_array, verbose=0)
        scores = np.asarray(raw_preds[0], dtype=np.float64)

        if scores.ndim != 1 or len(scores) != len(_class_names):
            raise ValueError("Unexpected prediction output shape.")

        if not np.all(np.isfinite(scores)):
            raise ValueError("Model returned non-finite prediction scores.")

        # Verify softmax contract: values in [0,1] and sum ~ 1
        if scores.min() < -0.01 or scores.max() > 1.01:
            raise ValueError(
                f"Prediction scores outside [0,1]: "
                f"min={scores.min():.4f} max={scores.max():.4f}"
            )

        pred_idx = int(np.argmax(scores))
        pred_class = _class_names[pred_idx]
        confidence = float(scores[pred_idx])   # raw probability [0, 1]

        all_scores = {
            name: round(float(s), 6)
            for name, s in zip(_class_names, scores)
        }   # raw probabilities [0, 1] -- frontend pct() converts to %

        return jsonify({
            "success": True,
            "predicted_class": pred_class,
            "confidence": round(confidence, 6),
            "note": (
                "confidence is the raw softmax score for this class "
                "and is not a calibrated probability"
            ),
            "all_scores": all_scores,
        }), 200

    except ValueError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 400

    except Exception:
        logger.exception("Disease prediction failed")
        return jsonify({
            "success": False,
            "error": "Prediction failed. Please try another image.",
        }), 500
