from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from typing import Any

def build_model() -> Any:
    """Return sklearn pipeline siap pakai."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

def train_model(model, X_train: pd.DataFrame, y_train: np.ndarray):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> str:
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

def predict_labels(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)