"""Feature definitions shared by training and inference (one source of truth)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Machine failure"
MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # label parts: NEVER features

TYPE_VALUES = ["L", "M", "H"]
CATEGORICAL = ["Type"]
RAW_NUMERIC = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
ENGINEERED = ["temp_diff", "power_w"]
NUMERIC = RAW_NUMERIC + ENGINEERED
FEATURES = CATEGORICAL + NUMERIC
REQUIRED_INPUT = CATEGORICAL + RAW_NUMERIC  # what an API caller must supply


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temp_diff"] = out["Process temperature"] - out["Air temperature"]
    # mechanical power in watts: torque (Nm) * angular speed (rad/s)
    out["power_w"] = out["Torque"] * out["Rotational speed"] * 2 * np.pi / 60
    return out


def make_xy(df: pd.DataFrame):
    """Return (X, y). y is None when the target column is absent (inference)."""
    missing = [c for c in REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = add_engineered_features(df)[FEATURES]
    y = df[TARGET] if TARGET in df.columns else None
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Unfitted preprocessor. It is fitted inside a Pipeline, on training data only."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            # handle_unknown="error": an invalid Type is rejected, not silently zeroed
            ("cat", OneHotEncoder(categories=[TYPE_VALUES], handle_unknown="error"), CATEGORICAL),
        ]
    )


if __name__ == "__main__":
    dev = pd.read_csv("data/dev.csv")
    X, y = make_xy(dev)
    print("X shape:", X.shape, "| y shape:", y.shape)
    print("Feature columns:", list(X.columns))
    print(X.head(3))
    print("Transformed shape:", build_preprocessor().fit_transform(X).shape)