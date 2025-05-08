from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import json
from datetime import datetime

# Load the model and MultiLabelBinarizer
model = joblib.load("packing_model.pkl")
mlb = joblib.load("packing_mlb.pkl")

# File for storing user feedback
FEEDBACK_FILE = "feedback_data.json"

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Allows requests from the React Native frontend

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Check for required fields
    required_fields = {"travel_type", "climate", "duration"}
    if not required_fields.issubset(data):
        return jsonify({"error": "Missing required fields"}), 400

    # Prepare data for prediction
    df = pd.DataFrame([{
        "travel_type": data["travel_type"],
        "climate": data["climate"],
        "duration": data["duration"]
    }])

    # Make prediction
    prediction = model.predict(df)
    items = mlb.inverse_transform(prediction)

    return jsonify({"recommended_items": list(items[0])})

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()

    # Check for required fields
    required_fields = {"travel_type", "climate", "duration", "items"}
    if not required_fields.issubset(data):
        return jsonify({"error": "Missing fields in feedback"}), 400

    # Load existing feedback or initialize
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = []

    # Append new feedback
    feedback_data.append(data)

    # Save feedback
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, ensure_ascii=False, indent=2)

    return jsonify({"message": "Feedback received"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
