import random
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib

# -------------------------------
# 1. Génération du dataset synthétique
# -------------------------------

types_voyage = ["plage", "montagne", "ville", "affaires"]
climats = ["chaud", "froid", "tempéré"]
objets_possibles = {
    "plage": ["maillot", "crème solaire", "lunettes", "short", "serviette"],
    "montagne": ["veste", "bottes", "gants", "bonnet", "sac de randonnée"],
    "ville": ["chaussures confort", "guide", "sac à dos", "appareil photo"],
    "affaires": ["costume", "ordinateur", "chargeur", "documents"]
}

def generer_exemple():
    type_v = random.choice(types_voyage)
    climat = random.choice(climats)
    duree = random.randint(2, 14)

    objets = set(objets_possibles.get(type_v, []))

    # Ajouter quelques objets génériques
    if duree > 5:
        objets.add("lessive")
    objets.add("brosse à dents")
    objets.add("chargeur")

    # Climat spécifique
    if climat == "froid":
        objets.add("manteau")
    elif climat == "chaud":
        objets.add("chapeau")

    return {
        "type_voyage": type_v,
        "climat": climat,
        "duree": duree,
        "objets": list(objets)
    }

# Créer 100 exemples
dataset = [generer_exemple() for _ in range(100)]
df = pd.DataFrame(dataset)

# -------------------------------
# 2. Préparation des données
# -------------------------------

X = df[["type_voyage", "climat", "duree"]]
y = df["objets"]

# Encodage des labels multi-objets
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y)

# Encodage des features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ["type_voyage", "climat"])
    ],
    remainder='passthrough'  # durée reste inchangée
)

# -------------------------------
# 3. Modèle
# -------------------------------

pipeline = Pipeline(steps=[
    ('preproc', preprocessor),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])

pipeline.fit(X, Y)

# -------------------------------
# 4. Exemple de prédiction
# -------------------------------

exemple = pd.DataFrame([{
    "type_voyage": "plage",
    "climat": "chaud",
    "duree": 7
}])

pred = pipeline.predict(exemple)
objets_predits = mlb.inverse_transform(pred)

print("Objets recommandés :", objets_predits[0])

# -------------------------------
# 5. Sauvegarde du modèle
# -------------------------------

joblib.dump(pipeline, "packing_model.pkl")
joblib.dump(mlb, "packing_mlb.pkl")
print("Modèle sauvegardé.")
