from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS  # Importer flask_cors

# Charger le modèle et le MultiLabelBinarizer
model = joblib.load("packing_model.pkl")
mlb = joblib.load("packing_mlb.pkl")

app = Flask(__name__)
CORS(app)  # Appliquer le CORS à l'application

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Vérification minimale
    required_fields = {"type_voyage", "climat", "duree"}
    if not required_fields.issubset(data):
        return jsonify({"error": "Champs manquants"}), 400

    # Formater les données pour le modèle
    df = pd.DataFrame([{
        "type_voyage": data["type_voyage"],
        "climat": data["climat"],
        "duree": data["duree"]
    }])

    # Prédiction
    prediction = model.predict(df)
    objets = mlb.inverse_transform(prediction)

    return jsonify({"objets_recommandes": list(objets[0])})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Modifié pour écouter sur toutes les interfaces
