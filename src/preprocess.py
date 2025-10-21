from pathlib import Path
import pandas as pd
import numpy as np

def load_data(path: str | Path = None):
    """Load CSV dari path atau default data folder."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "health_lifestyle_dataset.csv"
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Simple cleaning: drop exact-duplicate rows and fill small number of NA."""
    df = df.copy()
    df = df.drop_duplicates()
    # contoh filling sederhana (sesuaikan kebutuhan)
    df["age"] = df["age"].fillna(df["age"].median())
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df["daily_steps"] = df["daily_steps"].fillna(df["daily_steps"].median())
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature dataframe X siap dipakai model."""
    X = df[["age", "bmi", "daily_steps"]].copy()
    return X

def create_labels(df: pd.DataFrame) -> np.ndarray:
    """Return label y. Gunakan kolom ground-truth jika tersedia, kalau tidak buat pseudo-label."""
    if "risk" in df.columns:
        return df["risk"].astype(int).to_numpy()
    # pseudo-label (hanya contoh)
    return np.where((df["bmi"] > 25) & (df["daily_steps"] < 500), 1, 0)