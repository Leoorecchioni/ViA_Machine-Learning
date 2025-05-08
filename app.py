from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import json
from datetime import datetime

# Charger le modèle et le MultiLabelBinarizer
model = joblib.load("packing_model.pkl")
mlb = joblib.load("packing_mlb.pkl")

# Fichier pour stocker les retours utilisateurs
FEEDBACK_FILE = "feedback_data.jsonl"

# Initialisation de Flask
app = Flask(__name__)
CORS(app)  # Permet les requêtes depuis le frontend React Native

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Vérification des champs requis
    required_fields = {"type_voyage", "climat", "duree"}
    if not required_fields.issubset(data):
        return jsonify({"error": "Champs manquants"}), 400

    # Préparation des données pour le modèle
    df = pd.DataFrame([{
        "type_voyage": data["type_voyage"],
        "climat": data["climat"],
        "duree": data["duree"]
    }])

    # Prédiction
    prediction = model.predict(df)
    objets = mlb.inverse_transform(prediction)

    return jsonify({"objets_recommandes": list(objets[0])})

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()

    # Vérifier les champs
    required_fields = {"type_voyage", "climat", "duree", "objets"}
    if not required_fields.issubset(data):
        return jsonify({"error": "Champs manquants dans le feedback"}), 400

    # Charger les anciens feedbacks ou initialiser
    try:
        with open("feedback_data.json", "r", encoding="utf-8") as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = []

    # Ajouter le nouveau feedback
    feedback_data.append(data)

    # Sauvegarder
    with open("feedback_data.json", "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, ensure_ascii=False, indent=2)

    return jsonify({"message": "Feedback reçu"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
