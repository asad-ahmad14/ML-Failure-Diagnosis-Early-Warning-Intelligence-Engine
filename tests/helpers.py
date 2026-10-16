import pandas as pd

COLS = {
    "Air temperature": "air_temperature_k",
    "Process temperature": "process_temperature_k",
    "Rotational speed": "rotational_speed_rpm",
    "Torque": "torque_nm",
    "Tool wear": "tool_wear_min",
}

# Both payloads were checked against the running API:
# NORMAL gave warning_level "normal", LOWPOWER gave "high".
NORMAL = {
    "product_type": "M", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
    "rotational_speed_rpm": 1500, "torque_nm": 40, "tool_wear_min": 100,
}
LOWPOWER = {
    "product_type": "L", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
    "rotational_speed_rpm": 2800, "torque_nm": 5, "tool_wear_min": 50,
}


def payload_from_dev(row_id: int) -> dict:
    dev = pd.read_csv("data/dev.csv")
    row = dev[dev["row_id"] == row_id].iloc[0]
    out = {api: float(row[col]) for col, api in COLS.items()}
    out["product_type"] = str(row["Type"])
    return out