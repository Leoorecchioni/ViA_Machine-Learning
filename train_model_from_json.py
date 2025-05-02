import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Charger les données
with open("packing_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Créer un DataFrame
df = pd.DataFrame(data)

# Binariser les objets (multi-label)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["objets"])

# Encodage des colonnes catégorielles
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(), ["type_voyage", "climat"])
], remainder="passthrough")  # garde 'duree'

# Pipeline avec prétraitement + modèle
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Entraînement
pipeline.fit(df[["type_voyage", "climat", "duree"]], y)

# Sauvegarde du modèle et du MultiLabelBinarizer
joblib.dump(pipeline, "packing_model.pkl")
joblib.dump(mlb, "packing_mlb.pkl")

print("✅ Modèle entraîné et sauvegardé avec succès.")
