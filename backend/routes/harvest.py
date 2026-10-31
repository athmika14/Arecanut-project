from flask import Blueprint, request, jsonify

harvest_bp = Blueprint("harvest", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@harvest_bp.route("/predict", methods=["POST"])
def predict_harvest():
    """
    Endpoint: POST /api/harvest/predict
    Expects: multipart/form-data with an image file under the key 'image'
    Returns: harvest readiness result (stubbed until the ML model is ready)
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send the image under the key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, webp"}), 400

    # ---------------------------------------------------------------
    # ML MODEL PLACEHOLDER
    # The harvest readiness model has not been trained yet.
    # Once the model is ready, load it here and run inference.
    # ---------------------------------------------------------------
    return jsonify({
        "status": "model_not_ready",
        "message": "The harvest readiness model is not yet available. Please add the trained model to backend/models/harvest_model.h5 and update this route.",
        "filename": file.filename
    }), 503
