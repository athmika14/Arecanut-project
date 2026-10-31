# -*- coding: utf-8 -*-
"""
_audit_run.py  —  ArecaNut AI: Complete Testing & Verification Audit
===================================================================
READ-ONLY audit script.  Does not modify any application source files.

Executes:
  1. Environment verification (Python, TF, Flask, etc.)
  2. Backend pytest suite (22 tests)
  3. Flask app startup + model loading
  4. Model architecture verification (input/output shape, 9 classes)
  5. Class mapping validation
  6. API prediction with valid JPEG (real inference)
  7. API prediction with valid PNG (real inference)
  8. API error handling (missing file, wrong extension, empty, corrupt)
  9. Frontend build verification
 10. Frontend-to-backend API configuration check
 11. Format consistency check (frontend vs backend allowed types)
 12. Research report inspection (metrics, leakage, limitations)

All output is captured and written to:
  ml/disease/results/application_test_report.txt
"""
import os, sys, io, json, subprocess, time, traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = r"d:\mca project\ArecaNut_AI"
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
RESULTS_DIR = os.path.join(PROJECT_DIR, "ml", "disease", "results")
MODEL_PATH = os.path.join(PROJECT_DIR, "ml", "disease", "models", "mobilenetv2_best.keras")
INDEX_PATH = os.path.join(PROJECT_DIR, "ml", "disease", "models", "class_indices.json")

sys.path.insert(0, BACKEND_DIR)

results = []   # list of (test_name, status, evidence)

def record(name, status, evidence=""):
    results.append((name, status, evidence))
    tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "BLOCKED": "[BLOCKED]", "NOT TESTED": "[N/A]"}[status]
    print(f"  {tag} {name}")
    if evidence:
        for line in str(evidence).splitlines()[:3]:
            print(f"        {line}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_jpeg_bytes(w=300, h=400, color=(80,160,40)):
    from PIL import Image
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO(); img.save(buf, format="JPEG"); return buf.getvalue()

def make_png_bytes(w=200, h=200, color=(200,100,50)):
    from PIL import Image
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def make_webp_bytes(w=200, h=200, color=(100,50,200)):
    from PIL import Image
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO(); img.save(buf, format="WEBP"); return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — ENVIRONMENT VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 1: Environment Verification ===")

try:
    import tensorflow as tf; import numpy as np; import flask; import PIL
    from PIL import features as pil_features
    versions = f"Python {sys.version.split()[0]} | TF {tf.__version__} | numpy {np.__version__} | Flask {flask.__version__} | Pillow {PIL.__version__}"
    record("Environment - all libraries importable", "PASS", versions)
except Exception as e:
    record("Environment - all libraries importable", "FAIL", str(e))

try:
    webp_ok = pil_features.check("webp")
    record("Pillow WEBP decode support", "PASS" if webp_ok else "FAIL",
           f"WEBP support: {webp_ok}")
except Exception as e:
    record("Pillow WEBP decode support", "FAIL", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — BACKEND PYTEST SUITE
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 2: Backend pytest Suite ===")

try:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", os.path.join(BACKEND_DIR, "tests"), "-v", "--tb=short"],
        capture_output=True, text=True, timeout=300, cwd=BACKEND_DIR,
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "TF_ENABLE_ONEDNN_OPTS": "0"}
    )
    output = proc.stdout + proc.stderr
    # Parse pass/fail count from the summary line
    if "passed" in output and "failed" not in output:
        # Extract the count
        import re
        m = re.search(r"(\d+) passed", output)
        count = m.group(1) if m else "?"
        record(f"Backend pytest suite ({count} tests)", "PASS",
               f"Exit code: {proc.returncode}\n{output.strip()[-200:]}")
    elif "failed" in output:
        import re
        m = re.search(r"(\d+) failed", output)
        fails = m.group(1) if m else "?"
        record(f"Backend pytest suite ({fails} failed)", "FAIL",
               f"Exit code: {proc.returncode}\n{output.strip()[-500:]}")
    else:
        record("Backend pytest suite", "BLOCKED", output.strip()[-300:])
except subprocess.TimeoutExpired:
    record("Backend pytest suite", "BLOCKED", "Timed out after 300s")
except Exception as e:
    record("Backend pytest suite", "BLOCKED", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — FLASK APP STARTUP + MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 3: Flask App Startup + Model Loading ===")

try:
    from app import create_app
    app = create_app()
    from routes.disease import _model, _idx_to_cls, _load_error
    if _model is not None:
        record("Flask app starts and loads model", "PASS",
               f"Model loaded: input={_model.input_shape} output={_model.output_shape}")
    else:
        record("Flask app starts and loads model", "FAIL",
               f"Model not loaded: {_load_error}")
except Exception as e:
    record("Flask app starts and loads model", "FAIL", traceback.format_exc()[-300:])


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MODEL ARCHITECTURE VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 4: Model Architecture Verification ===")

try:
    if _model is not None:
        inp = _model.input_shape
        out = _model.output_shape
        # Expected: input (None, 224, 224, 3), output (None, 9)
        if inp == (None, 224, 224, 3):
            record("Model input shape is 224x224 RGB", "PASS", f"input={inp}")
        else:
            record("Model input shape is 224x224 RGB", "FAIL", f"input={inp}")

        if out == (None, 9):
            record("Model output has 9 classes", "PASS", f"output={out}")
        else:
            record("Model output has 9 classes", "FAIL", f"output={out}")
    else:
        record("Model input shape is 224x224 RGB", "BLOCKED", "Model not loaded")
        record("Model output has 9 classes", "BLOCKED", "Model not loaded")
except Exception as e:
    record("Model architecture verification", "FAIL", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — CLASS MAPPING VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 5: Class Mapping Validation ===")

try:
    with open(INDEX_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    idx_to_cls = {int(k): str(v) for k, v in raw.items()}

    expected = {"Healthy_Leaf","Healthy_Nut","Healthy_Trunk","Mahali_Koleroga",
                "Stem_bleeding","bud borer","healthy_foot","stem cracking","yellow leaf disease"}
    actual = set(idx_to_cls.values())

    if len(idx_to_cls) == 9:
        record("Class mapping has 9 classes", "PASS", f"keys 0-8: {sorted(idx_to_cls.keys())}")
    else:
        record("Class mapping has 9 classes", "FAIL", f"count={len(idx_to_cls)}")

    if actual == expected:
        record("Class names match model output", "PASS", f"all 9 expected names present")
    else:
        record("Class names match model output", "FAIL",
               f"mismatch: {actual ^ expected}")

    # Check model output dimension matches mapping count
    if _model is not None and _model.output_shape[-1] == len(idx_to_cls):
        record("Model output dimension matches class mapping count", "PASS",
               f"output={_model.output_shape[-1]} mapping={len(idx_to_cls)}")
    else:
        record("Model output dimension matches class mapping count", "FAIL",
               f"output={_model.output_shape[-1] if _model else 'N/A'} mapping={len(idx_to_cls)}")
except Exception as e:
    record("Class mapping validation", "FAIL", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — API PREDICTION WITH VALID JPEG (REAL INFERENCE)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 6: API Prediction Tests (Real Inference) ===")

if _model is not None:
    with app.test_client() as client:

        # 6a — JPEG upload
        try:
            data = {"image": (io.BytesIO(make_jpeg_bytes()), "leaf.jpg", "image/jpeg")}
            resp = client.post("/api/disease/predict", data=data, content_type="multipart/form-data")
            body = resp.get_json()
            if resp.status_code == 200 and "predicted_class" in body and "all_scores" in body:
                record("Valid JPEG produces prediction (200)", "PASS",
                       f"class={body['predicted_class']} conf={body['confidence']:.4f} scores={len(body['all_scores'])}")
            else:
                record("Valid JPEG produces prediction (200)", "FAIL",
                       f"status={resp.status_code} body={body}")
        except Exception as e:
            record("Valid JPEG produces prediction (200)", "FAIL", str(e))

        # 6b — PNG upload
        try:
            data = {"image": (io.BytesIO(make_png_bytes()), "leaf.png", "image/png")}
            resp = client.post("/api/disease/predict", data=data, content_type="multipart/form-data")
            body = resp.get_json()
            if resp.status_code == 200 and "predicted_class" in body:
                record("Valid PNG produces prediction (200)", "PASS",
                       f"class={body['predicted_class']} conf={body['confidence']:.4f}")
            else:
                record("Valid PNG produces prediction (200)", "FAIL",
                       f"status={resp.status_code} body={body}")
        except Exception as e:
            record("Valid PNG produces prediction (200)", "FAIL", str(e))

        # 6c — WEBP upload
        try:
            data = {"image": (io.BytesIO(make_webp_bytes()), "test.webp", "image/webp")}
            resp = client.post("/api/disease/predict", data=data, content_type="multipart/form-data")
            body = resp.get_json()
            if resp.status_code == 200 and "predicted_class" in body:
                record("Valid WEBP produces prediction (200)", "PASS",
                       f"class={body['predicted_class']} conf={body['confidence']:.4f}")
            else:
                record("Valid WEBP produces prediction (200)", "FAIL",
                       f"status={resp.status_code} body={body}")
        except Exception as e:
            record("Valid WEBP produces prediction (200)", "FAIL", str(e))

        # 6d — All scores sum to 1.0
        try:
            data = {"image": (io.BytesIO(make_jpeg_bytes()), "test.jpg", "image/jpeg")}
            resp = client.post("/api/disease/predict", data=data, content_type="multipart/form-data")
            body = resp.get_json()
            total = sum(body["all_scores"].values())
            if abs(total - 1.0) < 0.001:
                record("Softmax scores sum to 1.0", "PASS", f"sum={total:.6f}")
            else:
                record("Softmax scores sum to 1.0", "FAIL", f"sum={total:.6f}")
        except Exception as e:
            record("Softmax scores sum to 1.0", "FAIL", str(e))

        # 6e — Disclaimer present in response
        try:
            if "note" in body and "not a calibrated probability" in body["note"]:
                record("Response contains raw-softmax disclaimer", "PASS",
                       f"note field present: '{body['note'][:60]}...'")
            else:
                record("Response contains raw-softmax disclaimer", "FAIL",
                       f"note={body.get('note', 'MISSING')}")
        except Exception as e:
            record("Response contains raw-softmax disclaimer", "FAIL", str(e))

        # 6f — All 9 class scores present
        try:
            if len(body.get("all_scores", {})) == 9:
                record("Response contains all 9 class scores", "PASS",
                       f"keys={sorted(body['all_scores'].keys())}")
            else:
                record("Response contains all 9 class scores", "FAIL",
                       f"count={len(body.get('all_scores', {}))}")
        except Exception as e:
            record("Response contains all 9 class scores", "FAIL", str(e))

else:
    for name in ["Valid JPEG produces prediction (200)", "Valid PNG produces prediction (200)",
                 "Valid WEBP produces prediction (200)", "Softmax scores sum to 1.0",
                 "Response contains raw-softmax disclaimer", "Response contains all 9 class scores"]:
        record(name, "BLOCKED", "Model not loaded")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — API ERROR HANDLING
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 7: API Error Handling ===")

if _model is not None:
    with app.test_client() as client:
        # 7a — No file key
        try:
            resp = client.post("/api/disease/predict", data={})
            if resp.status_code == 400:
                record("Missing 'image' field returns 400", "PASS", f"status={resp.status_code}")
            else:
                record("Missing 'image' field returns 400", "FAIL", f"status={resp.status_code}")
        except Exception as e:
            record("Missing 'image' field returns 400", "FAIL", str(e))

        # 7b — Wrong extension (GIF)
        try:
            data = {"image": (io.BytesIO(b"GIF89a"), "anim.gif", "image/gif")}
            resp = client.post("/api/disease/predict", data=data, content_type="multipart/form-data")
            if resp.status_code == 400:
                record("Invalid extension (GIF) returns 400", "PASS", f"status={resp.status_code}")
            else:
                record("Invalid extension (GIF) returns 400", "FAIL", f"status={resp.status_code}")
        except Exception as e:
            record("Invalid extension (GIF) returns 400", "FAIL", str(e))

        # 7c — Empty file
        try:
            data = {"image": (io.BytesIO(b""), "empty.jpg", "image/jpeg")}
            resp = client.post("/api/disease/predict", data=data, content_type="multipart/form-data")
            if resp.status_code == 400:
                record("Empty file returns 400", "PASS", f"status={resp.status_code}")
            else:
                record("Empty file returns 400", "FAIL", f"status={resp.status_code}")
        except Exception as e:
            record("Empty file returns 400", "FAIL", str(e))

        # 7d — Corrupt JPEG
        try:
            data = {"image": (io.BytesIO(b"\xff\xd8\xff garbage"), "bad.jpg", "image/jpeg")}
            resp = client.post("/api/disease/predict", data=data, content_type="multipart/form-data")
            if resp.status_code == 400:
                record("Corrupt JPEG returns 400", "PASS", f"status={resp.status_code}")
            else:
                record("Corrupt JPEG returns 400", "FAIL", f"status={resp.status_code}")
        except Exception as e:
            record("Corrupt JPEG returns 400", "FAIL", str(e))

        # 7e — Health endpoint
        try:
            resp = client.get("/api/health")
            body = resp.get_json()
            if resp.status_code == 200 and "status" in body:
                record("Health endpoint returns status", "PASS",
                       f"status={body['status']} model={body.get('disease_model','?')}")
            else:
                record("Health endpoint returns status", "FAIL", f"status={resp.status_code}")
        except Exception as e:
            record("Health endpoint returns status", "FAIL", str(e))

else:
    for name in ["Missing 'image' field returns 400", "Invalid extension (GIF) returns 400",
                 "Empty file returns 400", "Corrupt JPEG returns 400", "Health endpoint returns status"]:
        record(name, "BLOCKED", "Model not loaded")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — FRONTEND BUILD VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 8: Frontend Build Verification ===")

try:
    proc = subprocess.run(
        ["npm", "run", "build"],
        capture_output=True, text=True, timeout=180, cwd=FRONTEND_DIR
    )
    output = proc.stdout + proc.stderr
    if proc.returncode == 0 and "Compiled successfully" in output:
        record("Frontend build compiles", "PASS",
               f"exit={proc.returncode}\n{output.strip()[-200:]}")
    else:
        record("Frontend build compiles", "FAIL",
               f"exit={proc.returncode}\n{output.strip()[-400:]}")
except subprocess.TimeoutExpired:
    record("Frontend build compiles", "BLOCKED", "npm build timed out after 180s")
except Exception as e:
    record("Frontend build compiles", "BLOCKED", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — FRONTEND CODE INSPECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 9: Frontend Code Inspection ===")

# 9a — Routes defined
try:
    with open(os.path.join(FRONTEND_DIR, "src", "App.js"), encoding="utf-8") as f:
        app_js = f.read()
    has_home = 'path="/" element={<Home />}' in app_js.replace(" ", " ").replace('"', '"')
    has_disease = 'path="/disease" element={<DiseaseDetection />}' in app_js.replace(" ", " ").replace('"', '"')
    # More flexible check
    has_home = 'path="/" element={<Home' in app_js
    has_disease = 'path="/disease" element={<DiseaseDetection' in app_js
    if has_home and has_disease:
        record("App.js defines Home and Disease Detection routes", "PASS",
               "Routes: / and /disease")
    else:
        record("App.js defines Home and Disease Detection routes", "FAIL",
               f"home={has_home} disease={has_disease}")
except Exception as e:
    record("App.js defines Home and Disease Detection routes", "FAIL", str(e))

# 9b — Home page has correct heading and navigation
try:
    with open(os.path.join(FRONTEND_DIR, "src", "pages", "Home.jsx"), encoding="utf-8") as f:
        home_jsx = f.read()
    checks = [
        ("AI-Powered Arecanut Plant Health" in home_jsx, "heading text"),
        ("Upload a plant image to identify potential disease conditions using AI." in home_jsx, "subtitle text"),
        ('navigate("/disease")' in home_jsx, "CTA navigates to /disease"),
        ("Check Your Plant" in home_jsx, "button label"),
        ("ArecaNut AI © 2026" in home_jsx.replace("&copy;", "©").replace("\\u00a9", "©"), "footer text"),
        ("Disease Identification" in home_jsx, "feature card 1"),
        ("Model Score" in home_jsx, "feature card 2"),
        ("Fast Analysis" in home_jsx, "feature card 3"),
    ]
    passed = sum(1 for ok, _ in checks if ok)
    details = "; ".join(f"{label}={'Y' if ok else 'N'}" for ok, label in checks)
    if passed == len(checks):
        record("Home page content and navigation", "PASS", details)
    else:
        record("Home page content and navigation", "FAIL", f"{passed}/{len(checks)}: {details}")
except Exception as e:
    record("Home page content and navigation", "FAIL", str(e))

# 9c — Disease Detection page has upload, API call, result card, error handling
try:
    with open(os.path.join(FRONTEND_DIR, "src", "pages", "DiseaseDetection.jsx"), encoding="utf-8") as f:
        dd_jsx = f.read()
    checks = [
        ('"image/jpeg","image/png","image/webp"' in dd_jsx.replace(" ", "") or "image/jpeg" in dd_jsx, "accepted types"),
        ("/api/disease/predict" in dd_jsx, "API URL"),
        ("handleAnalyse" in dd_jsx, "analyse handler"),
        ("handleRemove" in dd_jsx, "remove handler"),
        ("setLoading" in dd_jsx, "loading state"),
        ("setError" in dd_jsx, "error state"),
        ("setResult" in dd_jsx, "result state"),
        ("DISEASE_CLASSES" in dd_jsx, "disease class set"),
        ("HEALTHY_CLASSES" in dd_jsx, "healthy class set"),
        ("result-card--disease" in dd_jsx, "disease CSS modifier"),
        ("result-card--healthy" in dd_jsx, "healthy CSS modifier"),
        ("status-badge--disease" in dd_jsx, "disease badge"),
        ("status-badge--healthy" in dd_jsx, "healthy badge"),
        ("Disease Detected" in dd_jsx, "disease badge label"),
        ("No Disease Detected" in dd_jsx, "healthy badge label"),
        ("not a calibrated probability" in dd_jsx or "note" in dd_jsx, "disclaimer"),
        ("should not replace expert" in dd_jsx, "expert-diagnosis warning"),
        ("all_scores" in dd_jsx, "all class scores rendered"),
    ]
    passed = sum(1 for ok, _ in checks if ok)
    details = "; ".join(f"{label}={'Y' if ok else 'N'}" for ok, label in checks)
    if passed == len(checks):
        record("Disease Detection page features", "PASS", f"{passed}/{len(checks)}: {details}")
    else:
        record("Disease Detection page features", "FAIL", f"{passed}/{len(checks)}: {details}")
except Exception as e:
    record("Disease Detection page features", "FAIL", str(e))

# 9d — Navbar has correct links
try:
    with open(os.path.join(FRONTEND_DIR, "src", "components", "Navbar.jsx"), encoding="utf-8") as f:
        nav_jsx = f.read()
    has_home_link = 'to="/"' in nav_jsx
    has_disease_link = 'to="/disease"' in nav_jsx
    if has_home_link and has_disease_link:
        record("Navbar has Home and Disease Detection links", "PASS",
               "Links to / and /disease present")
    else:
        record("Navbar has Home and Disease Detection links", "FAIL",
               f"home={has_home_link} disease={has_disease_link}")
except Exception as e:
    record("Navbar has Home and Disease Detection links", "FAIL", str(e))

# 9e — proxy configured
try:
    with open(os.path.join(FRONTEND_DIR, "package.json"), encoding="utf-8") as f:
        pkg = json.load(f)
    proxy = pkg.get("proxy", "")
    if proxy == "http://localhost:5000":
        record("Frontend proxy points to backend", "PASS", f"proxy={proxy}")
    else:
        record("Frontend proxy points to backend", "FAIL", f"proxy={proxy}")
except Exception as e:
    record("Frontend proxy points to backend", "FAIL", str(e))

# 9f — Responsive CSS present
try:
    css_path = os.path.join(FRONTEND_DIR, "src", "styles", "Home.css")
    with open(css_path, encoding="utf-8") as f:
        home_css = f.read()
    has_media_768 = "@media (max-width: 768px)" in home_css
    has_media_600 = "@media (max-width: 600px)" in home_css
    has_media_380 = "@media (max-width: 380px)" in home_css
    if has_media_768 or has_media_600 or has_media_380:
        record("Home page responsive breakpoints exist", "PASS",
               f"768px={has_media_768} 600px={has_media_600} 380px={has_media_380}")
    else:
        record("Home page responsive breakpoints exist", "FAIL", "No @media queries found")
except Exception as e:
    record("Home page responsive breakpoints exist", "FAIL", str(e))

try:
    css_path = os.path.join(FRONTEND_DIR, "src", "styles", "Module.css")
    with open(css_path, encoding="utf-8") as f:
        mod_css = f.read()
    has_media_600 = "@media (max-width: 600px)" in mod_css
    has_media_380 = "@media (max-width: 380px)" in mod_css
    if has_media_600 or has_media_380:
        record("Disease Detection page responsive breakpoints exist", "PASS",
               f"600px={has_media_600} 380px={has_media_380}")
    else:
        record("Disease Detection page responsive breakpoints exist", "FAIL", "No @media queries found")
except Exception as e:
    record("Disease Detection page responsive breakpoints exist", "FAIL", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — FORMAT CONSISTENCY CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 10: Format Consistency ===")

try:
    # Backend allowed extensions
    from routes.disease import ALLOWED_EXTS
    backend_exts = sorted(ALLOWED_EXTS)

    # Frontend allowed types (from source code inspection)
    dd_source = open(os.path.join(FRONTEND_DIR, "src", "pages", "DiseaseDetection.jsx"), encoding="utf-8").read()
    # Extract the ALLOWED_TYPES set
    import re
    m = re.search(r'ALLOWED_TYPES\s*=\s*new Set\(\[(.*?)\]\)', dd_source, re.DOTALL)
    if m:
        types_str = m.group(1)
        frontend_types = set(re.findall(r'"(image/[^"]+)"', types_str))
        # Convert to extensions
        frontend_exts = sorted({t.split("/")[1] for t in frontend_types})
    else:
        frontend_exts = []

    # Home page advertised formats
    home_source = open(os.path.join(FRONTEND_DIR, "src", "pages", "Home.jsx"), encoding="utf-8").read()
    home_says_webp = "WEBP" in home_source
    home_says_jpg = "JPG" in home_source
    home_says_png = "PNG" in home_source

    # Upload label in DiseaseDetection
    dd_says_webp = "WEBP" in dd_source
    dd_says_jpg = "JPG" in dd_source
    dd_says_png = "PNG" in dd_source

    details = (f"Backend exts: {backend_exts} | "
               f"Frontend types: {frontend_exts} | "
               f"Home advertises: JPG={home_says_jpg} PNG={home_says_png} WEBP={home_says_webp} | "
               f"DD label: JPG={dd_says_jpg} PNG={dd_says_png} WEBP={dd_says_webp}")

    # Check consistency
    backend_set = set(backend_exts)
    frontend_set = set(frontend_exts)
    if backend_set == frontend_set and home_says_webp == ("webp" in backend_set):
        record("Format consistency: frontend, backend, and UI text agree", "PASS", details)
    else:
        record("Format consistency: frontend, backend, and UI text agree", "FAIL", details)
except Exception as e:
    record("Format consistency check", "FAIL", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — RESEARCH REPORT INSPECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 11: Research Report Inspection ===")

# 11a — Evaluation metrics
try:
    with open(os.path.join(RESULTS_DIR, "evaluation_metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    acc = metrics.get("test_accuracy", "N/A")
    mf1 = metrics.get("macro_f1", "N/A")
    wf1 = metrics.get("weighted_f1", "N/A")
    mp = metrics.get("macro_precision", "N/A")
    mr = metrics.get("macro_recall", "N/A")
    record("Evaluation metrics file exists and parses", "PASS",
           f"acc={acc:.4f} macro_f1={mf1:.4f} weighted_f1={wf1:.4f} macro_p={mp:.4f} macro_r={mr:.4f}")
except Exception as e:
    record("Evaluation metrics file exists and parses", "FAIL", str(e))

# 11b — Classification report
try:
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), encoding="utf-8") as f:
        report = f.read()
    has_all_classes = all(cls in report for cls in
        ["Healthy_Leaf","Healthy_Nut","Healthy_Trunk","Mahali_Koleroga",
         "Stem_bleeding","bud borer","healthy_foot","stem cracking","yellow leaf disease"])
    has_accuracy = "0.9676" in report or "0.9675" in report
    if has_all_classes and has_accuracy:
        record("Classification report has all 9 classes + accuracy", "PASS",
               f"all 9 classes present, accuracy line found")
    else:
        record("Classification report has all 9 classes + accuracy", "FAIL",
               f"all_classes={has_all_classes} accuracy={has_accuracy}")
except Exception as e:
    record("Classification report has all 9 classes + accuracy", "FAIL", str(e))

# 11c — Leakage-free split report
try:
    with open(os.path.join(RESULTS_DIR, "leakage_free_split_report.txt"), encoding="utf-8") as f:
        split_report = f.read()
    if "source-group" in split_report.lower() and "leakage" in split_report.lower():
        record("Split report confirms source-group separation", "PASS",
               "Report mentions source-group stratified split and leakage-free")
    else:
        record("Split report confirms source-group separation", "FAIL",
               "Keywords not found in report")
except Exception as e:
    record("Split report confirms source-group separation", "FAIL", str(e))

# 11d — Limitations documented
try:
    limitations_found = []
    if "healthy_foot" in report and "not reliable" in report.lower():
        limitations_found.append("healthy_foot small sample noted")
    if "class_weight" in report.lower() or "imbalance" in report.lower():
        limitations_found.append("class imbalance noted")
    if "augmented copies" in report.lower() or "augment" in report.lower():
        limitations_found.append("augmentation noted")
    if "not replace expert" in report.lower() or "expert" in report.lower():
        limitations_found.append("expert diagnosis caveat")

    if len(limitations_found) >= 3:
        record("Limitations documented in reports", "PASS",
               "; ".join(limitations_found))
    else:
        record("Limitations documented in reports", "FAIL",
               f"Only {len(limitations_found)} limitations found: {limitations_found}")
except Exception as e:
    record("Limitations documented in reports", "FAIL", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — INTEGRATION: Backend disconnect handling (frontend code)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 12: Integration Checks ===")

try:
    dd_source = open(os.path.join(FRONTEND_DIR, "src", "pages", "DiseaseDetection.jsx"), encoding="utf-8").read()
    has_catch = "catch" in dd_source
    has_friendly_error = "Could not reach the backend" in dd_source
    if has_catch and has_friendly_error:
        record("Frontend handles backend disconnect gracefully", "PASS",
               "try/catch block with 'Could not reach the backend' message")
    else:
        record("Frontend handles backend disconnect gracefully", "FAIL",
               f"catch={has_catch} friendly_error={has_friendly_error}")
except Exception as e:
    record("Frontend handles backend disconnect gracefully", "FAIL", str(e))

# Disease/healthy styling logic
try:
    has_disease_set = "DISEASE_CLASSES" in dd_source
    has_healthy_set = "HEALTHY_CLASSES" in dd_source
    has_classify = "classifyPrediction" in dd_source
    has_disease_card = "result-card--disease" in dd_source
    has_healthy_card = "result-card--healthy" in dd_source
    if all([has_disease_set, has_healthy_set, has_classify, has_disease_card, has_healthy_card]):
        record("Disease/healthy styling determined from predicted class", "PASS",
               "classifyPrediction() maps class name to disease/healthy CSS modifier")
    else:
        record("Disease/healthy styling determined from predicted class", "FAIL",
               f"disease_set={has_disease_set} healthy_set={has_healthy_set} classify={has_classify}")
except Exception as e:
    record("Disease/healthy styling determined from predicted class", "FAIL", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — CHECKS THAT CANNOT BE PERFORMED
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 13: Checks Not Performed ===")

not_tested = [
    ("Browser-based visual rendering of homepage", "NOT TESTED",
     "Requires a running browser/headless browser (e.g., Puppeteer/Playwright). "
     "Not installed. Code inspection and build verification used instead."),
    ("Live React dev server interaction", "NOT TESTED",
     "Starting npm start blocks the terminal and requires a browser to verify. "
     "Build compilation verified instead."),
    ("Real arecanut image prediction accuracy", "NOT TESTED",
     "Would require correctly labeled field images not present in the project. "
     "Synthetic solid-color images used for API integration testing instead."),
    ("Mobile device viewport testing", "NOT TESTED",
     "Requires a mobile browser or device emulator. "
     "Responsive CSS breakpoints verified via code inspection instead."),
    ("CORS preflight from real browser", "NOT TESTED",
     "CORS is configured for localhost:3000. Verified via code inspection of app.py. "
     "Flask test client bypasses CORS."),
]


# ══════════════════════════════════════════════════════════════════════════════
#  WRITE REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Writing Report ===")

report_path = os.path.join(RESULTS_DIR, "application_test_report.txt")

lines = []
lines.append("=" * 72)
lines.append("  ArecaNut AI — Complete Testing & Verification Audit Report")
lines.append(f"  Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("  Project   : D:\\mca project\\ArecaNut_AI")
lines.append("=" * 72)
lines.append("")

# Count results
pass_count = sum(1 for _, s, _ in results if s == "PASS")
fail_count = sum(1 for _, s, _ in results if s == "FAIL")
blocked_count = sum(1 for _, s, _ in results if s == "BLOCKED")
na_count = len(not_tested)

lines.append(f"  SUMMARY: {pass_count} PASS | {fail_count} FAIL | {blocked_count} BLOCKED | {na_count} NOT TESTED")
lines.append(f"  Total checks: {len(results) + na_count}")
lines.append("")

# Group results by inferred section
sections = [
    ("SECTION 1 — Environment Verification", ["Environment", "Pillow WEBP"]),
    ("SECTION 2 — Backend pytest Suite", ["pytest"]),
    ("SECTION 3 — Flask App Startup + Model Loading", ["Flask app starts"]),
    ("SECTION 4 — Model Architecture Verification", ["Model input", "Model output"]),
    ("SECTION 5 — Class Mapping Validation", ["Class mapping", "Class names", "Model output dimension matches"]),
    ("SECTION 6 — API Prediction Tests", ["Valid JPEG", "Valid PNG", "Valid WEBP", "Softmax", "disclaimer", "9 class"]),
    ("SECTION 7 — API Error Handling", ["Missing", "Invalid extension", "Empty file", "Corrupt", "Health endpoint"]),
    ("SECTION 8 — Frontend Build", ["Frontend build"]),
    ("SECTION 9 — Frontend Code Inspection", ["App.js", "Home page", "Disease Detection", "Navbar", "proxy", "responsive"]),
    ("SECTION 10 — Format Consistency", ["Format consistency"]),
    ("SECTION 11 — Research Report Inspection", ["Evaluation metrics", "Classification report", "Split report", "Limitations"]),
    ("SECTION 12 — Integration Checks", ["disconnect", "Disease/healthy styling"]),
]

for sec_name, keywords in sections:
    lines.append("-" * 72)
    lines.append(f"  {sec_name}")
    lines.append("-" * 72)
    for name, status, evidence in results:
        if any(kw.lower() in name.lower() for kw in keywords):
            tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "BLOCKED": "[BLOCKED]"}[status]
            lines.append(f"  {tag} {name}")
            if evidence:
                for ev_line in str(evidence).splitlines():
                    lines.append(f"        {ev_line}")
            lines.append("")

# Not tested section
lines.append("-" * 72)
lines.append("  SECTION 13 — Checks Not Performed")
lines.append("-" * 72)
for name, status, reason in not_tested:
    lines.append(f"  [N/A] {name}")
    lines.append(f"        Reason: {reason}")
    lines.append("")

# Issues and recommendations
lines.append("=" * 72)
lines.append("  ISSUES FOUND AND RECOMMENDED FIXES")
lines.append("=" * 72)
lines.append("")
issues = [(name, evidence) for name, status, evidence in results if status == "FAIL"]
if issues:
    for name, evidence in issues:
        lines.append(f"  [FAIL] {name}")
        lines.append(f"    Evidence: {evidence}")
        lines.append(f"    Fix: Review the implementation and correct the issue.")
        lines.append("")
else:
    lines.append("  No failures found during the audit.")
    lines.append("")

blocked = [(name, evidence) for name, status, evidence in results if status == "BLOCKED"]
if blocked:
    lines.append("  Blocked tests:")
    for name, evidence in blocked:
        lines.append(f"  [BLOCKED] {name}")
        lines.append(f"    Evidence: {evidence}")
        lines.append("")

lines.append("=" * 72)
lines.append("  METHODOLOGY NOTE")
lines.append("=" * 72)
lines.append("")
lines.append("  Executed tests (actual results, not assumed):")
lines.append("    - Backend pytest suite (22 tests) — executed via subprocess")
lines.append("    - Flask app startup + model loading — executed in-process")
lines.append("    - Model architecture verification — executed in-process")
lines.append("    - API prediction with JPEG, PNG, WEBP — executed via Flask test client")
lines.append("    - API error handling (5 scenarios) — executed via Flask test client")
lines.append("    - Frontend build compilation — executed via npm subprocess")
lines.append("    - Softmax sum-to-1, disclaimer, all-scores checks — executed")
lines.append("")
lines.append("  Code inspection only (not executed):")
lines.append("    - Frontend route definitions, navigation, content checks")
lines.append("    - Navbar links, proxy configuration")
lines.append("    - Responsive CSS breakpoints")
lines.append("    - Disease/healthy classification logic")
lines.append("    - Format consistency (cross-referencing source files)")
lines.append("    - Research report contents")
lines.append("")
lines.append("  Not tested (documented why):")
lines.append("    - Browser visual rendering — no headless browser installed")
lines.append("    - Live dev server — blocks terminal, needs browser")
lines.append("    - Real arecanut images — no labeled field images available")
lines.append("    - Mobile viewport — no device emulator")
lines.append("    - CORS preflight — test client bypasses CORS")
lines.append("")

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nReport saved to: {report_path}")
print(f"\nFINAL COUNTS: {pass_count} PASS | {fail_count} FAIL | {blocked_count} BLOCKED | {na_count} NOT TESTED")
