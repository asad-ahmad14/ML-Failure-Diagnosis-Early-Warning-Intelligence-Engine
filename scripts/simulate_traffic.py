"""Send real dev-set rows to the running API so the dashboard has data to show.
These rows were used for training, so scores look better than they would on unseen data."""
import argparse

import httpx
import pandas as pd

COLS = {
    "Air temperature": "air_temperature_k",
    "Process temperature": "process_temperature_k",
    "Rotational speed": "rotational_speed_rpm",
    "Torque": "torque_nm",
    "Tool wear": "tool_wear_min",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dev = pd.read_csv("data/dev.csv").sample(args.n, random_state=args.seed)
    payloads = []
    for _, row in dev.iterrows():
        item = {api: float(row[col]) for col, api in COLS.items()}
        item["product_type"] = str(row["Type"])
        payloads.append(item)

    counts: dict[str, int] = {}
    for i in range(0, len(payloads), 100):  # the API's batch limit is 100
        r = httpx.post(
            f"{args.api}/predict/batch",
            json={"readings": payloads[i : i + 100]},
            timeout=120,
        )
        r.raise_for_status()
        for res in r.json()["results"]:
            counts[res["warning_level"]] = counts.get(res["warning_level"], 0) + 1
    print(f"sent {len(payloads)} readings; warning levels: {counts}")


if __name__ == "__main__":
    main()