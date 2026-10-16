"""Dev-only analysis (no test data): confirm where misses come from and
compare candidate cutoffs for the 'high' level."""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_curve

from ml.features import MODE_COLUMNS, TARGET
from ml.train_final import MODEL_NAME, metrics_at

ARTIFACTS = Path("artifacts")


def main() -> None:
    dev = pd.read_csv("data/dev.csv")
    oof = pd.read_csv(ARTIFACTS / "oof_scores.csv")
    assert oof["row_id"].equals(dev["row_id"]), "OOF rows not aligned with dev rows"
    y = dev[TARGET].to_numpy()
    scores = oof[MODEL_NAME].to_numpy()
    with open(ARTIFACTS / "model_metadata.json") as f:
        elevated = json.load(f)["thresholds"]["elevated"]

    missed = (y == 1) & (scores < elevated)
    print(f"Missed failures at 'elevated' cutoff: {int(missed.sum())} of {int(y.sum())}")
    for mode in MODE_COLUMNS:
        n = int((missed & (dev[mode].to_numpy() == 1)).sum())
        print(f"  {mode}-labelled among missed: {n}")
    no_mode = dev[MODE_COLUMNS].sum(axis=1).to_numpy() == 0
    print(f"  no mode label among missed: {int((missed & no_mode).sum())}")

    precision, _, thr = precision_recall_curve(y, scores)
    p = precision[:-1]
    rows = []
    for target in [0.90, 0.95, 0.97, 0.99]:
        ok = thr[p >= target]
        if len(ok) == 0:
            print(f"precision >= {target}: not reachable")
            continue
        rows.append({"target_precision": target, **metrics_at(y, scores, float(ok.min()))})
    print("\nCandidate 'high' cutoffs (OOF dev):")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()