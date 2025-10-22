import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# === Load Dataset ===
df = pd.read_csv("health_lifestyle_dataset.csv")

# Hapus kolom id jika ada
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Encode gender
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Pisahkan fitur dan label
X = df.drop(columns=['disease_risk'])
y = df['disease_risk']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Simpan model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model berhasil dilatih dan disimpan sebagai model.pkl")
