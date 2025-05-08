import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Chemins des fichiers
MAIN_DATA_FILE = "packing_data.json"
FEEDBACK_FILE = "feedback_data.jsonl"
MODEL_FILE = "packing_model.pkl"
MLB_FILE = "packing_mlb.pkl"

# 1. Charger les données d'entraînement principales
with open(MAIN_DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Charger les retours utilisateurs s'ils existent
if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        feedback_lines = f.readlines()
        feedback_data = [json.loads(line.strip()) for line in feedback_lines]
    # Fusionner
    data += [{
        "type_voyage": d["type_voyage"],
        "climat": d["climat"],
        "duree": d["duree"],
        "objets": d["objets_recommandes"]
    } for d in feedback_data]
    print(f"✅ {len(feedback_data)} retours utilisateurs intégrés.")
else:
    print("⚠️ Aucun feedback trouvé. Entraînement avec les données initiales.")

# 3. Convertir en DataFrame
df = pd.DataFrame(data)

# 4. Multi-label binarization
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["objets"])

# 5. Pipeline avec OneHotEncoder + RandomForest
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(), ["type_voyage", "climat"])
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Entraînement
pipeline.fit(df[["type_voyage", "climat", "duree"]], y)

# 7. Sauvegarder
joblib.dump(pipeline, MODEL_FILE)
joblib.dump(mlb, MLB_FILE)

print("✅ Modèle réentraîné avec succès avec les nouvelles données !")
