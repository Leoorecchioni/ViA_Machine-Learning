import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# File paths
MAIN_DATA_FILE = "packing_data.json"
FEEDBACK_FILE = "feedback_data.jsonl"
MODEL_FILE = "packing_model.pkl"
MLB_FILE = "packing_mlb.pkl"

# 1. Load main training data
with open(MAIN_DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Load user feedback if it exists
if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        feedback_lines = f.readlines()
        feedback_data = [json.loads(line.strip()) for line in feedback_lines]
    # Merge data with translated keys
    data += [{
        "travel_type": d["type_voyage"],
        "climate": d["climat"],
        "duration": d["duree"],
        "items": d["objets_recommandes"]
    } for d in feedback_data]
    print(f"✅ {len(feedback_data)} user feedback entries integrated.")
else:
    print("⚠️ No feedback found. Training with initial data only.")

# 3. Convert to DataFrame
df = pd.DataFrame(data)

# 4. Multi-label binarization
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["items"])

# 5. Pipeline with OneHotEncoder + RandomForest
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(), ["travel_type", "climate"])
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Training
pipeline.fit(df[["travel_type", "climate", "duration"]], y)

# 7. Save
joblib.dump(pipeline, MODEL_FILE)
joblib.dump(mlb, MLB_FILE)

print("✅ Model retrained successfully with the new data!")
