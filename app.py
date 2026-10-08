from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained employee promotion model
model = joblib.load("employee_promotion_model.pkl")


@app.route("/")
def home():
    return "Employee Promotion Prediction API is running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)[0]

    return jsonify({
        "promotion_prediction": int(prediction)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
