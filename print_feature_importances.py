import os
import pickle
import numpy as np
import pandas as pd

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan di {model_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Urutan fitur sesuai train_model.py (tanpa kolom id dan label)
features = [
    "age","gender","bmi","daily_steps","sleep_hours","water_intake_l",
    "calories_consumed","smoker","alcohol","resting_hr","systolic_bp",
    "diastolic_bp","cholesterol","family_history"
]

# Jika model punya atribut feature_importances_
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": features, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))
else:
    print("Model tidak memiliki atribut feature_importances_.")