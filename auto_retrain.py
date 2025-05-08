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
        print(f"📁 Backup directory created: {BACKUP_DIR}")

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

    # Rename keys to match English format
    def transform(entry):
        return {
            "travel_type": entry.get("travel_type", entry.get("travel_type")),
            "climate": entry.get("climate", entry.get("climate")),
            "duration": entry.get("duration", entry.get("duration")),
            "items": entry.get("items", entry.get("items"))
        }

    return [transform(d) for d in original + feedback], [transform(fb) for fb in feedback]

def backup_feedback(feedback):
    if not feedback:
        return
    ensure_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = os.path.join(BACKUP_DIR, f"feedback_backup_{timestamp}.json")
    with open(backup_filename, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2, ensure_ascii=False)
    print(f"💾 Backup saved: {backup_filename}")

def train_model(data, feedback):
    if not data:
        print("❌ No dataset found.")
        return

    df = pd.DataFrame(data)

    # Binarization
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["items"])

    # Preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(), ["travel_type", "climate"])
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(df[["travel_type", "climate", "duration"]], y)

    joblib.dump(pipeline, "packing_model.pkl")
    joblib.dump(mlb, "packing_mlb.pkl")
    print("✅ Model retrained and saved.")

    # Backup + cleanup
    backup_feedback(feedback)
    with open("feedback_data.json", "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)
    print("🧹 Feedback processed and file cleared.")

if __name__ == "__main__":
    while True:
        print("🔁 Automatic training in progress...")
        all_data, feedback_only = load_data()
        train_model(all_data, feedback_only)
        time.sleep(30)  # pause between trainings
