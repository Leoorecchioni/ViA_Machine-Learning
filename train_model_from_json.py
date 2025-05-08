import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
with open("packing_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Create a DataFrame
df = pd.DataFrame(data)

# Binarize items (multi-label)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["objets"])

# Encode categorical columns
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(), ["type_voyage", "climat"])
], remainder="passthrough")  # keep 'duree' as is

# Pipeline with preprocessing + model
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Training
pipeline.fit(df[["type_voyage", "climat", "duree"]], y)

# Save the model and MultiLabelBinarizer
joblib.dump(pipeline, "packing_model.pkl")
joblib.dump(mlb, "packing_mlb.pkl")

print("✅ Model trained and saved successfully.")
