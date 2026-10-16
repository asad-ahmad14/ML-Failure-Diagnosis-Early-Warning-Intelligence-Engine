"""Exploratory checks on the raw AI4I data. Prints facts; changes nothing."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/ai4i2020_raw.csv")
TARGET = "Machine failure"
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUMERIC = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    df = pd.read_csv(RAW_PATH)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nTotal missing values:", int(df.isna().sum().sum()))
    print("Fully duplicated rows:", int(df.duplicated().sum()))
    print("Duplicate feature rows:", int(df.duplicated(subset=["Type"] + NUMERIC).sum()))

    print("\nType counts:\n", df["Type"].value_counts())
    print("\nFailure rate by Type:\n", df.groupby("Type")[TARGET].agg(["count", "sum", "mean"]))

    print("\nFailure mode counts:\n", df[MODES].sum())
    n_modes = df[MODES].sum(axis=1)
    print("\nRows by number of active failure modes:\n", n_modes.value_counts().sort_index())

    any_mode = (n_modes > 0).astype(int)
    print("\nLabel consistency (rows = Machine failure, columns = any mode active):")
    print(pd.crosstab(df[TARGET], any_mode, rownames=["Machine failure"], colnames=["any mode"]))
    print("\nRNF vs Machine failure:")
    print(pd.crosstab(df["RNF"], df[TARGET], rownames=["RNF"], colnames=["Machine failure"]))

    # Engineered candidates (checked here, not yet used by any model)
    df["temp_diff"] = df["Process temperature"] - df["Air temperature"]
    df["power_w"] = df["Torque"] * df["Rotational speed"] * 2 * np.pi / 60

    cols = NUMERIC + ["temp_diff", "power_w"]
    print("\nValue ranges:\n", df[cols].agg(["min", "max"]).T)
    print("\nMean of each column by Machine failure (0 = normal, 1 = failure):")
    print(df.groupby(TARGET)[cols].mean().T)


if __name__ == "__main__":
    main()