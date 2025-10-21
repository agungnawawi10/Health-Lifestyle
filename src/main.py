import argparse
from preprocess import load_data, clean_data, create_features, create_labels
from model import build_model, train_model, evaluate_model, predict_labels
import numpy as np
import pandas as pd

def main(data_path: str | None = None, no_ml: bool = False):
    df = load_data(data_path)
    df = clean_data(df)

    X = create_features(df)
    y = create_labels(df)

    unique = np.unique(y)
    print("Label distribusi:", dict(zip(unique.tolist(), np.bincount(y) if y.dtype.kind in "iu" else np.unique(y, return_counts=True)[1].tolist())))

    if no_ml or unique.size < 2:
        # fallback: simpan label langsung sebagai predicted result
        df["predicted_risk_ml"] = np.where(y == 1, "High", "Low")
        print("ML training skipped (no valid classes or --no-ml).")
    else:
        model = build_model()
        # split dengan stratify bila perlu (import di sini untuk keep imports minimal)
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = train_model(model, X_train, y_train)
        print("=== Evaluasi Model ===")
        print(evaluate_model(model, X_test, y_test))
        df["predicted_risk_ml"] = np.where(predict_labels(model, X) == 1, "High", "Low")

    print(df[["age", "bmi", "daily_steps", "predicted_risk_ml"]].head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to dataset CSV", default=None)
    parser.add_argument("--no-ml", help="Skip ML training (use rule labels)", action="store_true")
    args = parser.parse_args()
    main(data_path=args.data, no_ml=args.no_ml)

