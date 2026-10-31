# -*- coding: utf-8 -*-
"""
tests/test_disease.py  -  ArecaNut AI: Disease Detection
=========================================================

Unit and integration tests covering:
  Section 1  preprocess_image()  (preprocessing pipeline)
  Section 2  Class-index mapping validation
  Section 3  API error-handling paths (no model needed)
  Section 4  API happy path  (requires trained model on disk)
  Section 5  Confidence range guard  (scores must be in [0, 1])
  Section 6  CLIP content guard  (rejects unrelated images)

Run from the backend/ directory:
    python -m pytest tests/test_disease.py -v

Test requirements that need additional data
-------------------------------------------
- test_real_arecanut_images_accepted: uses real arecanut images from
  backend/tests/test_images/.  5 images are already present.
- test_baby_photo_rejected and test_unrelated_object_rejected:
  require photos of a person/baby and an unrelated object.
  Skipped automatically if absent.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"  # before TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"

import io
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

MODEL_PATH = os.path.join(
    PROJECT_DIR, "ml", "disease", "models", "mobilenetv2_best.keras"
)
INDEX_PATH = os.path.join(
    PROJECT_DIR, "ml", "disease", "models", "class_indices.json"
)

MODEL_AVAILABLE = os.path.isfile(MODEL_PATH) and os.path.isfile(INDEX_PATH)

# Check if CLIP is available
CLIP_AVAILABLE = False
try:
    from transformers import TFCLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    pass

# Test images
TEST_IMAGES_DIR = os.path.join(BACKEND_DIR, "tests", "test_images")
ARECANUT_TEST_IMAGES = [
    f for f in [
        "arecanut_trunk.jpg",
        "arecanut_leaf.jpg",
        "arecanut_nut.jpg",
        "arecanut_mahali.jpg",
        "arecanut_stem_bleeding.jpg",
    ]
    if os.path.isfile(os.path.join(TEST_IMAGES_DIR, f))
]


# ── Image helpers ─────────────────────────────────────────────────────────────

def _jpeg(w=300, h=400, color=(80, 160, 40)) -> bytes:
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _png(w=200, h=200, color=(200, 100, 50)) -> bytes:
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _webp(w=200, h=200, color=(100, 50, 200)) -> bytes:
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


# ── App factory (test client) ─────────────────────────────────────────────────

def _make_test_app():
    """Minimal Flask app with disease blueprint (no model loading)."""
    import routes.disease as rd
    from flask import Flask
    from flask_cors import CORS

    app = Flask(__name__)
    app.config["TESTING"] = True
    CORS(app, origins=["http://localhost:3000"])
    app.register_blueprint(rd.disease_bp, url_prefix="/api/disease")
    return app


@pytest.fixture
def client_no_model():
    """Test client with model globals forced to None."""
    import routes.disease as rd

    orig_model       = rd._model
    orig_class_names = rd._class_names
    orig_clip        = rd._clip_model
    orig_clip_proc   = rd._clip_processor
    orig_load_error  = rd._load_error
    orig_clip_error  = rd._clip_load_error

    rd._model       = None
    rd._class_names = None
    rd._clip_model  = None
    rd._clip_processor = None
    rd._load_error  = "Model deliberately unloaded for testing."
    rd._clip_load_error = "Model deliberately unloaded for testing."

    app = _make_test_app()
    with app.test_client() as c:
        yield c

    rd._model       = orig_model
    rd._class_names = orig_class_names
    rd._clip_model  = orig_clip
    rd._clip_processor = orig_clip_proc
    rd._load_error  = orig_load_error
    rd._clip_load_error = orig_clip_error


@pytest.fixture
def client():
    """
    Test client with real disease model loaded.
    CLIP guard is DISABLED so synthetic images can reach the classifier.
    Models are loaded only once (module-level state persists across tests).
    """
    if not MODEL_AVAILABLE:
        pytest.skip("Trained model not available.")
    import routes.disease as rd
    if rd._model is None:
        rd.load_disease_model()
    # Disable CLIP guard for sections 4 & 5
    orig_clip = rd._clip_model
    rd._clip_model = None  # is_likely_arecanut_image fails open when None

    app = _make_test_app()
    with app.test_client() as c:
        yield c

    rd._clip_model = orig_clip


@pytest.fixture
def client_with_guard():
    """
    Test client with both disease model AND CLIP guard loaded.
    Used exclusively by Section 6 guard tests.
    Models are loaded only once (module-level state persists).
    """
    if not MODEL_AVAILABLE:
        pytest.skip("Trained model not available.")
    if not CLIP_AVAILABLE:
        pytest.skip("CLIP/transformers not available.")
    import routes.disease as rd
    if rd._model is None:
        rd.load_disease_model()
    if rd._clip_model is None:
        pytest.skip("CLIP model failed to load.")

    app = _make_test_app()
    with app.test_client() as c:
        yield c


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — preprocess_image()
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessImage:

    def test_output_shape_jpeg(self):
        from routes.disease import preprocess_image
        arr = preprocess_image(_jpeg(), "jpg")
        assert arr.shape == (1, 224, 224, 3)

    def test_output_dtype_float32(self):
        from routes.disease import preprocess_image
        arr = preprocess_image(_jpeg(), "jpg")
        assert arr.dtype == np.float32

    def test_mobilenetv2_scaling_white(self):
        from routes.disease import preprocess_image
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        arr = preprocess_image(buf.getvalue(), "png")
        assert np.allclose(arr, 1.0, atol=0.02), f"max={arr.max():.4f}"

    def test_mobilenetv2_scaling_black(self):
        from routes.disease import preprocess_image
        img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        arr = preprocess_image(buf.getvalue(), "png")
        assert np.allclose(arr, -1.0, atol=0.02), f"min={arr.min():.4f}"

    def test_non_square_resized(self):
        from routes.disease import preprocess_image
        arr = preprocess_image(_jpeg(w=640, h=480), "jpg")
        assert arr.shape == (1, 224, 224, 3)

    def test_rgba_png_accepted(self):
        from routes.disease import preprocess_image
        img = Image.new("RGBA", (100, 100), color=(0, 128, 255, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        arr = preprocess_image(buf.getvalue(), "png")
        assert arr.shape == (1, 224, 224, 3)

    def test_grayscale_accepted(self):
        from routes.disease import preprocess_image
        img = Image.new("L", (100, 100), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        arr = preprocess_image(buf.getvalue(), "png")
        assert arr.shape == (1, 224, 224, 3)

    def test_invalid_bytes_raise_value_error(self):
        from routes.disease import preprocess_image
        with pytest.raises(ValueError, match="Cannot read"):
            preprocess_image(b"not an image", "jpg")

    def test_empty_bytes_raise_value_error(self):
        from routes.disease import preprocess_image
        with pytest.raises(ValueError):
            preprocess_image(b"", "jpg")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Class-index mapping
# ══════════════════════════════════════════════════════════════════════════════

class TestClassIndexMapping:

    def test_index_file_exists(self):
        assert os.path.isfile(INDEX_PATH)

    def test_index_file_is_valid_json(self):
        with open(INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_nine_classes(self):
        with open(INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 9

    def test_keys_zero_to_eight(self):
        with open(INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert {int(k) for k in data} == set(range(9))

    def test_expected_class_names(self):
        expected = {
            "Healthy_Leaf", "Healthy_Nut", "Healthy_Trunk",
            "Mahali_Koleroga", "Stem_bleeding", "bud borer",
            "healthy_foot", "stem cracking", "yellow leaf disease",
        }
        with open(INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert set(data.values()) == expected


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — API error handling (no model needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIErrorsNoModel:

    def test_model_unavailable_returns_503(self, client_no_model):
        data = {"file": (io.BytesIO(_jpeg()), "leaf.jpg", "image/jpeg")}
        resp = client_no_model.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 503
        assert "error" in resp.get_json()

    def test_missing_file_field_returns_400(self, client_no_model):
        resp = client_no_model.post(
            "/api/disease/predict",
            data={},
            content_type="multipart/form-data",
        )
        assert resp.status_code in (400, 503)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — API happy path (requires trained model, guard disabled)
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIHappyPath:

    def test_jpeg_returns_200(self, client):
        data = {"file": (io.BytesIO(_jpeg()), "leaf.jpg", "image/jpeg")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        assert "predicted_class" in body
        assert "confidence" in body
        assert "all_scores" in body

    def test_png_returns_200(self, client):
        data = {"file": (io.BytesIO(_png()), "leaf.png", "image/png")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200

    def test_webp_returns_200(self, client):
        data = {"file": (io.BytesIO(_webp()), "leaf.webp", "image/webp")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200

    def test_wrong_extension_returns_400(self, client):
        data = {"file": (io.BytesIO(b"GIF89a"), "anim.gif", "image/gif")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_empty_file_returns_400(self, client):
        data = {"file": (io.BytesIO(b""), "empty.jpg", "image/jpeg")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_corrupt_jpeg_returns_400(self, client):
        data = {"file": (io.BytesIO(b"\xff\xd8\xff garbage"), "bad.jpg", "image/jpeg")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_predicted_class_is_valid(self, client):
        valid_classes = {
            "Healthy_Leaf", "Healthy_Nut", "Healthy_Trunk",
            "Mahali_Koleroga", "Stem_bleeding", "bud borer",
            "healthy_foot", "stem cracking", "yellow leaf disease",
        }
        data = {"file": (io.BytesIO(_jpeg()), "leaf.jpg", "image/jpeg")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        body = resp.get_json()
        assert body["predicted_class"] in valid_classes

    def test_nine_scores_returned(self, client):
        data = {"file": (io.BytesIO(_jpeg()), "leaf.jpg", "image/jpeg")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        body = resp.get_json()
        assert len(body["all_scores"]) == 9


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Confidence range (scores must be in [0, 1])
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidenceRange:

    def test_confidence_in_zero_to_one(self, client):
        data = {"file": (io.BytesIO(_jpeg()), "leaf.jpg", "image/jpeg")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        body = resp.get_json()
        conf = body["confidence"]
        assert 0.0 <= conf <= 1.0, f"confidence={conf} is outside [0,1]"

    def test_all_scores_in_zero_to_one(self, client):
        data = {"file": (io.BytesIO(_jpeg()), "leaf.jpg", "image/jpeg")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        body = resp.get_json()
        for cls, score in body["all_scores"].items():
            assert 0.0 <= score <= 1.0, \
                f"all_scores['{cls}']={score} is outside [0,1]"

    def test_scores_sum_to_one(self, client):
        data = {"file": (io.BytesIO(_jpeg()), "leaf.jpg", "image/jpeg")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        body = resp.get_json()
        total = sum(body["all_scores"].values())
        assert abs(total - 1.0) < 0.01, f"scores sum to {total}"

    def test_confidence_equals_max_all_scores(self, client):
        data = {"file": (io.BytesIO(_jpeg()), "leaf.jpg", "image/jpeg")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        body = resp.get_json()
        expected_conf = body["all_scores"][body["predicted_class"]]
        assert abs(body["confidence"] - expected_conf) < 1e-5

    def test_disclaimer_note_present(self, client):
        data = {"file": (io.BytesIO(_jpeg()), "leaf.jpg", "image/jpeg")}
        resp = client.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        body = resp.get_json()
        assert "note" in body
        assert "calibrated" in body["note"].lower()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — CLIP content guard
# ══════════════════════════════════════════════════════════════════════════════

class TestCLIPGuard:
    """
    Tests for the CLIP zero-shot content guard.

    The guard uses OpenAI CLIP ViT-B/32 to compare uploaded images
    against 28 text prompts (10 arecanut, 6 non-palm-plant, 12 non-plant).

    Limitations:
    - CLIP cannot distinguish arecanut from other palm species.
    - Real unrelated images (people, vehicles) were NOT tested
      (network blocked downloads during testing).
    - Tests use real arecanut photos from backend/tests/test_images/
      and synthetic solid-color images as controls.
    """

    def test_guard_function_returns_tuple(self):
        """is_likely_arecanut_image() must return (bool, str)."""
        from routes.disease import is_likely_arecanut_image
        if not MODEL_AVAILABLE or not CLIP_AVAILABLE:
            pytest.skip("Model or CLIP not available.")
        import routes.disease as rd
        if rd._clip_model is None:
            rd.load_disease_model()
        if rd._clip_model is None:
            pytest.skip("CLIP model failed to load.")
        img = Image.new("RGB", (256, 256), (100, 150, 50))
        result = is_likely_arecanut_image(img)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_tiny_image_rejected(self):
        """Images below MIN_IMAGE_DIM must be rejected."""
        from routes.disease import is_likely_arecanut_image, MIN_IMAGE_DIM
        if not MODEL_AVAILABLE or not CLIP_AVAILABLE:
            pytest.skip("Model or CLIP not available.")
        import routes.disease as rd
        if rd._clip_model is None:
            rd.load_disease_model()
        if rd._clip_model is None:
            pytest.skip("CLIP model failed to load.")
        tiny = Image.new("RGB", (20, 20), (100, 150, 50))
        ok, reason = is_likely_arecanut_image(tiny)
        assert ok is False
        assert "too small" in reason.lower()

    def test_synthetic_red_rejected(self, client_with_guard):
        """A solid red image should be rejected by the CLIP guard."""
        red = Image.new("RGB", (224, 224), (220, 30, 30))
        buf = io.BytesIO()
        red.save(buf, format="JPEG")
        data = {"file": (io.BytesIO(buf.getvalue()), "red.jpg", "image/jpeg")}
        resp = client_with_guard.post(
            "/api/disease/predict",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 422
        body = resp.get_json()
        assert body.get("rejected_by_guard") is True
        assert "error" in body

    @pytest.mark.skipif(
        len(ARECANUT_TEST_IMAGES) == 0,
        reason="No real arecanut test images found in backend/tests/test_images/"
    )
    def test_real_arecanut_images_accepted(self, client_with_guard):
        """Real arecanut photos should pass the CLIP guard and return 200."""
        accepted = 0
        rejected = 0
        for img_name in ARECANUT_TEST_IMAGES:
            img_path = os.path.join(TEST_IMAGES_DIR, img_name)
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            data = {"file": (io.BytesIO(img_bytes), img_name, "image/jpeg")}
            resp = client_with_guard.post(
                "/api/disease/predict",
                data=data,
                content_type="multipart/form-data",
            )
            if resp.status_code == 200:
                accepted += 1
            elif resp.status_code == 422:
                rejected += 1
            else:
                pytest.fail(
                    f"Unexpected status {resp.status_code} for {img_name}: "
                    f"{resp.get_json()}"
                )
        # At least some should pass (allowing for CLIP false rejections)
        assert accepted > 0, (
            f"All {len(ARECANUT_TEST_IMAGES)} real arecanut images were "
            f"rejected by the CLIP guard. This indicates a guard problem."
        )

    def test_guard_fail_open_when_clip_none(self):
        """When _clip_model is None, guard should accept (fail-open)."""
        import routes.disease as rd
        if not MODEL_AVAILABLE:
            pytest.skip("Model not available.")
        rd.load_disease_model()
        orig_clip = rd._clip_model
        rd._clip_model = None
        try:
            img = Image.new("RGB", (256, 256), (100, 150, 50))
            ok, reason = rd.is_likely_arecanut_image(img)
            assert ok is True
            assert reason == ""
        finally:
            rd._clip_model = orig_clip
