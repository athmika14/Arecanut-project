import logging
from flask import Flask
from flask_cors import CORS
from config import Config
from routes.disease import disease_bp, load_disease_model
from routes.harvest import harvest_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Allow requests from the React dev server
    CORS(app, origins=["http://localhost:3000"])

    # Register blueprints (independent modules)
    # disease_bp uses url_prefix="/api/disease" so that the route
    # decorator "/predict" inside disease.py resolves to the correct
    # URL /api/disease/predict without duplication.
    app.register_blueprint(disease_bp, url_prefix="/api/disease")
    app.register_blueprint(harvest_bp, url_prefix="/api/harvest")

    # Load the disease model once at startup so every request reuses it
    with app.app_context():
        load_disease_model()

    @app.route("/api/health")
    def health_check():
        from routes.disease import (
            _model, _load_error,
            _clip_model, _clip_load_error,
        )
        return {
            "status": "ArecaNut AI backend is running",
            "disease_model": (
                "loaded" if _model is not None
                else f"unavailable: {_load_error}"
            ),
            "clip_guard": (
                "loaded" if _clip_model is not None
                else f"unavailable: {_clip_load_error}"
            ),
        }, 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
