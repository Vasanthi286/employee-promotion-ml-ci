
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Employee Promotion model
model = joblib.load("employee_promotion_model.pkl")


@app.route("/", methods=["GET"])
def home():
    return "Employee Promotion Prediction API is running", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data:
        return jsonify({
            "error": "Please provide employee details in JSON format"
        }), 400

    try:
        # Convert employee details into a DataFrame
        input_data = pd.DataFrame([data])

        # Predict employee promotion
        prediction = model.predict(input_data)[0]

        return jsonify({
            "promotion_prediction": int(prediction)
        }), 200

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
