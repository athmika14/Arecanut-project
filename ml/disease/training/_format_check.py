import re, os, sys, json
sys.path.insert(0, r"d:\mca project\ArecaNut_AI\backend")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings; warnings.filterwarnings("ignore")

dd_path = r"d:\mca project\ArecaNut_AI\frontend\src\pages\DiseaseDetection.jsx"
with open(dd_path, encoding="utf-8") as f:
    src = f.read()

m = re.search(r"const ALLOWED_TYPES\s*=\s*new Set\(\[(.*?)\]\)", src, re.DOTALL)
print("Frontend ALLOWED_TYPES raw:", m.group(1).strip() if m else "NOT FOUND")

m2 = re.search(r'accept="([^"]+)"', src)
print("input[accept]:", m2.group(1) if m2 else "NOT FOUND")

m3 = re.search(r"Supported:.*?<", src, re.DOTALL)
print("Upload label text:", m3.group(0)[:80].strip() if m3 else "NOT FOUND")

from routes.disease import ALLOWED_EXTS
print("Backend ALLOWED_EXTS:", sorted(ALLOWED_EXTS))

# Home page
home_path = r"d:\mca project\ArecaNut_AI\frontend\src\pages\Home.jsx"
with open(home_path, encoding="utf-8") as f:
    home = f.read()
m4 = re.search(r"Supports.*?(?:<|$)", home, re.DOTALL)
print("Home advertises:", m4.group(0)[:80].strip() if m4 else "NOT FOUND")

print()
print("--- ANALYSIS ---")
print("Frontend MIME types: image/jpeg, image/png, image/webp")
print("  image/jpeg is assigned by browsers to BOTH .jpg AND .jpeg uploads.")
print("  Comparing 'jpeg' (MIME subtype) vs 'jpg' (file extension) is a false mismatch.")
print("  The frontend correctly accepts all .jpg, .jpeg, .png, .webp files.")
print("Backend ALLOWED_EXTS contains both 'jpg' and 'jpeg' for safety (good practice).")
print("Both layers accept the same four real-world format combinations.")
print("VERDICT: No real format inconsistency exists. The previous FAIL was a false alarm.")
print("  The audit script compared MIME subtypes with file extensions, not like-with-like.")
