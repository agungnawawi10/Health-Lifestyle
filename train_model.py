import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import pickle
import os

# === Load Dataset ===
data_path = os.path.join("data", "health_lifestyle_dataset.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Hapus kolom id jika ada
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Encode gender
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

# Check label distribution
print(f"Label distribution:\n{df['disease_risk'].value_counts()}")

# Pisahkan fitur dan label
X = df.drop(columns=["disease_risk"])
y = df["disease_risk"]

# Split dataset with stratification to handle class imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with class_weight to handle imbalance
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)
model.fit(X_train, y_train)

# === Model Evaluation ===
# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Probabilities for AUC
y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

# Training metrics
train_acc = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

# Test metrics
test_acc = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print("\n=== MODEL PERFORMANCE ===")
print(f"Training Accuracy: {train_acc:.3f}")
print(f"Training AUC: {train_auc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")
print(f"Test AUC: {test_auc:.3f}")

print("\n=== TEST SET DETAILED METRICS ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE (Top 10) ===")
print(feature_importance.head(10).to_string(index=False))

# Simpan model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model berhasil dilatih dan disimpan sebagai model.pkl")
print(f"📊 Model dapat memprediksi dengan akurasi {test_acc:.1%} dan AUC {test_auc:.3f}")