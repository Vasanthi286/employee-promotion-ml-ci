
import logging
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify, g

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Locate the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    BASE_DIR / "employee_promotion_model.pkl",
    BASE_DIR.parent / "Experiment-9-GitHub-Actions"
    / "employee_promotion_model.pkl",
]

MODEL_PATH = next(
    (path for path in MODEL_PATHS if path.is_file()),
    None
)

if MODEL_PATH is None:
    raise FileNotFoundError(
        "employee_promotion_model.pkl not found. "
        "Place it in this folder or Experiment-9-GitHub-Actions."
    )

model = joblib.load(MODEL_PATH)
logger.info("Employee Promotion model loaded from %s", MODEL_PATH)


@app.before_request
def log_request_start():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()

    logger.info(
        "Request started | id=%s method=%s path=%s",
        g.request_id, request.method, request.path
    )


@app.after_request
def log_request_end(response):
    elapsed = time.time() - getattr(g, "start_time", time.time())
    request_id = getattr(g, "request_id", "unknown")

    logger.info(
        "Request completed | id=%s status=%s duration=%.3fs",
        request_id, response.status_code, elapsed
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.route("/", methods=["GET"])
def home():
    logger.info("Homepage accessed")
    return jsonify({
        "message": "Employee Promotion Prediction API is running"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    request_id = getattr(g, "request_id", "unknown")

    if not isinstance(data, dict) or not data:
        logger.warning("Invalid prediction request | id=%s", request_id)
        return jsonify({
            "error": "A non-empty JSON object is required",
            "request_id": request_id
        }), 400

    try:
        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)[0]

        logger.info("Prediction successful | id=%s", request_id)
        return jsonify({
            "promotion_prediction": int(prediction),
            "request_id": request_id
        }), 200

    except Exception:
        logger.exception("Prediction failed | id=%s", request_id)
        return jsonify({
            "error": "Prediction failed. Check application logs.",
            "request_id": request_id
        }), 400


@app.errorhandler(404)
def not_found(error):
    logger.warning(
        "Route not found | path=%s id=%s",
        request.path, getattr(g, "request_id", "unknown")
    )
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting Employee Promotion Prediction API")
    app.run(host="0.0.0.0", port=5000)

