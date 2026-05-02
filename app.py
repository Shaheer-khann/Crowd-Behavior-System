"""
Crowd Behavior Analysis — Flask Backend API
=============================================
Phase 1: Accepts video uploads via /predict, preprocesses using OpenCV,
runs inference with the CNN-LSTM model (best_model.h5), and returns
a JSON prediction.

Model Architecture: MobileNetV2 (frozen) → GlobalAveragePooling2D → LSTM(64) → Dense(128) → Dense(4, softmax)
Input Tensor Shape: (1, 16, 112, 112, 3)
Classes: normal, panic, aggressive, idle
"""

import os
import uuid
import traceback
import logging
from collections import Counter

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 16
IMG_SIZE = (112, 112)                         # (width, height) for cv2.resize
CLASSES = ["normal", "panic", "aggressive", "idle"]
MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024        # 100 MB upload cap
STRIDE = int(os.environ.get("STRIDE", 8))     # frames to advance between windows
TARGET_FPS = int(os.environ.get("TARGET_FPS", 8))  # max FPS to sample during extraction
MAX_WINDOW_DETAILS = int(os.environ.get("MAX_WINDOW_DETAILS", 50))  # cap JSON payload size

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)                                     # allow cross-origin for frontend
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading (done once at startup)
# ---------------------------------------------------------------------------
model = None


def get_model():
    """Lazy-load and cache the Keras model."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Place best_model.h5 in the project root or set the "
                "MODEL_PATH environment variable."
            )
        logger.info("Loading model from %s …", MODEL_PATH)
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    """Return True if the file extension is in the allow-list."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_frames(video_path: str) -> list[np.ndarray]:
    """
    Read frames from a video file sampled at TARGET_FPS, resize to IMG_SIZE
    and normalise pixel values to [0, 1].  Returns a list of np arrays shaped
    (H, W, 3).  Sampling at TARGET_FPS (default 8) reduces memory usage and
    inference count for high-framerate drone footage without losing accuracy.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # How many native frames to skip between each kept frame
    frame_interval = max(1, round(native_fps / TARGET_FPS))
    logger.info(
        "Video FPS: %.1f — keeping 1 frame every %d (target %d FPS)",
        native_fps, frame_interval, TARGET_FPS,
    )

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


def predict_on_frames(frames: list[np.ndarray]) -> dict:
    """
    Run a sliding-window prediction over the extracted frames.

    For every window of SEQUENCE_LENGTH consecutive frames the model produces
    a softmax vector; we aggregate by majority vote and also return per-window
    details.

    Returns a dict ready for JSON serialisation.
    """
    mdl = get_model()

    if len(frames) < SEQUENCE_LENGTH:
        return {
            "success": False,
            "error": (
                f"Video is too short. Need at least {SEQUENCE_LENGTH} frames "
                f"but only got {len(frames)}."
            ),
        }

    window_results = []
    all_labels: list[str] = []

    window_starts = range(0, len(frames) - SEQUENCE_LENGTH + 1, STRIDE)
    logger.info(f"Running inference on {len(window_starts)} windows (stride={STRIDE}).")

    for start in window_starts:
        window = frames[start : start + SEQUENCE_LENGTH]
        input_tensor = np.expand_dims(np.array(window), axis=0)  # (1, 16, 112, 112, 3)

        prediction = mdl.predict(input_tensor, verbose=0)[0]     # (4,)
        class_idx = int(np.argmax(prediction))
        class_label = CLASSES[class_idx]
        confidence = float(prediction[class_idx])

        window_results.append(
            {
                "window_start_frame": start,
                "window_end_frame": start + SEQUENCE_LENGTH - 1,
                "predicted_class": class_label,
                "confidence": round(confidence, 4),
                "probabilities": {
                    CLASSES[i]: round(float(prediction[i]), 4)
                    for i in range(len(CLASSES))
                },
            }
        )
        all_labels.append(class_label)

    # Majority vote
    counts = Counter(all_labels)
    dominant_label, dominant_count = counts.most_common(1)[0]

    # Ensure every class is represented in the summary (0 if never predicted)
    full_summary = {cls: counts.get(cls, 0) for cls in CLASSES}

    # Trim window_details to avoid sending huge JSON payloads
    trimmed_details = window_results[:MAX_WINDOW_DETAILS]
    details_truncated = len(window_results) > MAX_WINDOW_DETAILS

    return {
        "success": True,
        "total_frames_extracted": len(frames),
        "total_windows_analysed": len(window_results),
        "dominant_prediction": dominant_label,
        "dominant_count": dominant_count,
        "prediction_summary": full_summary,
        "window_details": trimmed_details,
        "window_details_truncated": details_truncated,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Serve the frontend dashboard."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Health-check / API status endpoint."""
    return jsonify({
        "status": "online",
        "service": "Crowd Behavior Analysis API",
        "model": MODEL_PATH,
        "classes": CLASSES,
        "expected_input": f"Video file via POST /predict (min {SEQUENCE_LENGTH} frames)",
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a video file upload, preprocess it, run model inference,
    and return the JSON prediction result.
    """
    # --- validate request ---------------------------------------------------
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No 'video' file part in request."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        }), 400

    # --- save uploaded file temporarily -------------------------------------
    ext = file.filename.rsplit(".", 1)[1].lower()
    temp_filename = f"{uuid.uuid4().hex}.{ext}"
    temp_path = None  # initialise so finally block never raises NameError

    try:
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_path)
        logger.info("Saved upload to %s", temp_path)

        # --- extract frames -------------------------------------------------
        frames = extract_frames(temp_path)
        logger.info("Extracted %d frames from uploaded video.", len(frames))

        # --- run prediction --------------------------------------------------
        result = predict_on_frames(frames)

        # Free frame memory immediately after inference
        del frames

        status_code = 200 if result.get("success") else 422
        return jsonify(result), status_code

    except Exception as exc:
        logger.error("Prediction failed: %s", traceback.format_exc())
        return jsonify({"success": False, "error": str(exc)}), 500

    finally:
        # Always clean up the temp file — even if an exception was raised
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("Cleaned up temp file %s", temp_path)
            except OSError as rm_err:
                logger.warning("Could not delete temp file %s: %s", temp_path, rm_err)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Pre-load the model so the first request isn't slow
    try:
        get_model()
    except FileNotFoundError as e:
        logger.warning("⚠️  %s — server will start, but /predict will fail until the model is available.", e)

    # Select server mode: development (Flask debug) or production (Waitress)
    env = os.environ.get("FLASK_ENV", "development").lower()

    if env == "production":
        from waitress import serve
        PORT = int(os.environ.get("PORT", 5000))
        logger.info("🚀 Starting production server (Waitress) on port %d …", PORT)
        serve(app, host="0.0.0.0", port=PORT, threads=4)
    else:
        logger.info("🔧 Starting development server (Flask debug) …")
        app.run(host="0.0.0.0", port=5000, debug=True)
