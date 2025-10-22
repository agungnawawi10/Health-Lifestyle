import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

project_root = os.path.dirname(__file__)
model_path = os.path.join(project_root, "model.pkl")
data_path = os.path.join(project_root, "data", "health_lifestyle_dataset.csv")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data not found: {data_path}")

# Load
with open(model_path, "rb") as f:
    model = pickle.load(f)
df = pd.read_csv(data_path)

# Prepare X, y same as training
if 'id' in df.columns:
    df = df.drop(columns=['id'])
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
X = df.drop(columns=['disease_risk'])
y = df['disease_risk']

print("Label distribution:")
print(y.value_counts(dropna=False))
print("\nFeature columns (order):")
print(X.columns.tolist())

# Basic per-label statistics
print("\nFeature means by label:")
print(X.groupby(y).mean().T)

# Show examples of label==1 and label==0
print("\nExamples with label==1 (up to 5):")
print(df[y==1].head()[X.columns.tolist() + ['disease_risk']])
print("\nExamples with label==0 (up to 5):")
print(df[y==0].head()[X.columns.tolist() + ['disease_risk']])

# Model predictions on training data (sanity)
try:
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(f"\nModel accuracy on dataset: {acc:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
except Exception as e:
    print("Error when predicting on dataset:", e)

# Average predicted probability for class 1 grouped by true label
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X)[:, 1]
    print("\nMean predicted P(class=1) for true label groups:")
    print(pd.DataFrame({"true": y, "p1": probs}).groupby("true").mean())

# Test the problematic sample (edit values if perlu)
sample = {
    "age": 70,
    "gender": "Male",
    "bmi": 38.5,
    "daily_steps": 1500,
    "sleep_hours": 4.0,
    "water_intake_l": 1.0,
    "calories_consumed": 3800,
    "smoker": 1,
    "alcohol": 1,
    "resting_hr": 95,
    "systolic_bp": 170,
    "diastolic_bp": 110,
    "cholesterol": 320,
    "family_history": 1
}
# encode and order same as training
s_gender = 1 if str(sample["gender"]).lower() == "female" else 0
sample_vec = np.array([[sample["age"], s_gender, sample["bmi"], sample["daily_steps"],
                        sample["sleep_hours"], sample["water_intake_l"], sample["calories_consumed"],
                        sample["smoker"], sample["alcohol"], sample["resting_hr"],
                        sample["systolic_bp"], sample["diastolic_bp"], sample["cholesterol"],
                        sample["family_history"]]], dtype=float)
print("\nSample feature vector:", sample_vec.flatten().tolist())

if hasattr(model, "predict_proba"):
    sp = model.predict_proba(sample_vec)[0]
    print("Sample predict_proba (class0, class1):", sp.tolist())
pred = model.predict(sample_vec)[0]
print("Sample predicted label:", pred)