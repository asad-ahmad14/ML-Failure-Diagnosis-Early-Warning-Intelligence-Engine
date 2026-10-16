"""Create the stratified dev/test split. The test set is touched once, at the end."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.features import TARGET

SEED = 42
RAW_PATH = Path("data/ai4i2020_raw.csv")
DEV_PATH = Path("data/dev.csv")
TEST_PATH = Path("data/test.csv")


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    df.insert(0, "row_id", range(len(df)))  # stable id for error analysis

    dev, test = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=SEED
    )
    assert set(dev["row_id"]).isdisjoint(set(test["row_id"])), "split overlap!"

    dev.sort_values("row_id").to_csv(DEV_PATH, index=False)
    test.sort_values("row_id").to_csv(TEST_PATH, index=False)

    for name, part in [("dev", dev), ("test", test)]:
        print(f"{name}: rows={len(part)} failures={int(part[TARGET].sum())} "
              f"rate={part[TARGET].mean():.4f}")
    print("total failures:", int(dev[TARGET].sum() + test[TARGET].sum()))


if __name__ == "__main__":
    main()