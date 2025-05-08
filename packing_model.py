import random
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib

# -------------------------------
# 1. Synthetic dataset generation
# -------------------------------

travel_types = ["beach", "mountain", "city", "business"]
climates = ["hot", "cold", "temperate"]
possible_items = {
    "beach": ["swimsuit", "sunscreen", "sunglasses", "shorts", "towel"],
    "mountain": ["jacket", "boots", "gloves", "hat", "hiking bag"],
    "city": ["comfortable shoes", "guidebook", "backpack", "camera"],
    "business": ["suit", "laptop", "charger", "documents"]
}

def generate_example():
    travel_type = random.choice(travel_types)
    climate = random.choice(climates)
    duration = random.randint(2, 14)

    items = set(possible_items.get(travel_type, []))

    # Add some generic items
    if duration > 5:
        items.add("laundry detergent")
    items.add("toothbrush")
    items.add("charger")

    # Climate-specific items
    if climate == "cold":
        items.add("coat")
    elif climate == "hot":
        items.add("hat")

    return {
        "travel_type": travel_type,
        "climate": climate,
        "duration": duration,
        "items": list(items)
    }

# Create 100 examples
dataset = [generate_example() for _ in range(100)]
df = pd.DataFrame(dataset)

# -------------------------------
# 2. Data preparation
# -------------------------------

X = df[["travel_type", "climate", "duration"]]
y = df["items"]

# Multi-label encoding
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y)

# Feature encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ["travel_type", "climate"])
    ],
    remainder='passthrough'  # duration remains unchanged
)

# -------------------------------
# 3. Model
# -------------------------------

pipeline = Pipeline(steps=[
    ('preproc', preprocessor),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])

pipeline.fit(X, Y)

# -------------------------------
# 4. Prediction example
# -------------------------------

example = pd.DataFrame([{
    "travel_type": "beach",
    "climate": "hot",
    "duration": 7
}])

pred = pipeline.predict(example)
predicted_items = mlb.inverse_transform(pred)

print("Recommended items:", predicted_items[0])

# -------------------------------
# 5. Model saving
# -------------------------------

joblib.dump(pipeline, "packing_model.pkl")
joblib.dump(mlb, "packing_mlb.pkl")
print("Model saved.")
