from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_PATH = Path("data/ai4i2020_raw.csv")


def download() -> pd.DataFrame:
    ds = fetch_ucirepo(id=601)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    print("Columns received:", list(df.columns))
    RAW_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    return df


def summarize(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Missing values per column:\n", df.isna().sum())
    print("Dtypes:\n", df.dtypes)
    target = "Machine failure"
    if target in df.columns:
        print("Failure count:", int(df[target].sum()))
        print("Failure rate: %.4f" % df[target].mean())
    print(df.describe().T)


if __name__ == "__main__":
    summarize(download())