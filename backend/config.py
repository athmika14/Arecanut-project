import os

class Config:
    # Maximum upload size: 16 MB
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # Folder where uploaded images are temporarily stored
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")

    # Allowed image extensions
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

    # Secret key (override via environment variable in production)
    SECRET_KEY = os.environ.get("SECRET_KEY", "arecanut-ai-dev-secret")
