import json
import os
import time
import joblib
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

BACKUP_DIR = "backups"

def ensure_backup_dir():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"📁 Dossier de backup créé : {BACKUP_DIR}")

def load_data():
    try:
        with open("packing_data.json", "r", encoding="utf-8") as f:
            original = json.load(f)
    except FileNotFoundError:
        original = []

    try:
        with open("feedback_data.json", "r", encoding="utf-8") as f:
            feedback = json.load(f)
    except FileNotFoundError:
        feedback = []

    return original + feedback, feedback

def backup_feedback(feedback):
    if not feedback:
        return
    ensure_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = os.path.join(BACKUP_DIR, f"feedback_backup_{timestamp}.json")
    with open(backup_filename, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2, ensure_ascii=False)
    print(f"💾 Backup enregistré : {backup_filename}")

def train_model(data, feedback):
    if not data:
        print("❌ Aucun jeu de données trouvé.")
        return

    df = pd.DataFrame(data)

    # Binarisation
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["objets"])

    # Prétraitement
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(), ["type_voyage", "climat"])
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(df[["type_voyage", "climat", "duree"]], y)

    joblib.dump(pipeline, "packing_model.pkl")
    joblib.dump(mlb, "packing_mlb.pkl")
    print("✅ Modèle réentraîné et sauvegardé.")

    # Backup + nettoyage
    backup_feedback(feedback)
    with open("feedback_data.json", "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)
    print("🧹 Feedback traité et fichier vidé.")

if __name__ == "__main__":
    while True:
        print("🔁 Entraînement automatique en cours...")
        all_data, feedback_only = load_data()
        train_model(all_data, feedback_only)
        time.sleep(30)  # pause entre chaque entraînement
