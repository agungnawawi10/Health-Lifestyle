import os
import pickle
import pandas as pd
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
data_path = os.path.join(os.path.dirname(__file__), "data", "health_lifestyle_dataset.csv")

with open(model_path, "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(data_path)
if "id" in df.columns:
    df = df.drop(columns=["id"])
if "disease_risk" in df.columns:
    X = df.drop(columns=["disease_risk"])
else:
    X = df.copy()

# Encode gender same as training
X['gender'] = X['gender'].map({'Male': 0, 'Female': 1})

features = X.columns.tolist()

# Optional: install shap with 'pip install shap' before running
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)  # shap_values[1] for class 1 in binary

# Mean absolute SHAP per fitur (global importance)
mean_abs_shap = np.abs(shap_values).mean(axis=1).mean(axis=0) if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)
imp_df = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs_shap})
print(imp_df.sort_values("mean_abs_shap", ascending=False).to_string(index=False))

# For visual: shap.summary_plot(shap_values, X)  # requires matplotlib in environment