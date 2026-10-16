from ml.features import FEATURES
from tests.helpers import payload_from_dev

import pandas as pd


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_predict_normal_training_row(client):
    r = client.post("/predict", json=payload_from_dev(0))
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["model_level"] == "normal"
    assert len(body["evidence"]) == 3
    assert {e["feature"] for e in body["evidence"]} <= set(FEATURES)


def test_predict_failure_training_row(client):
    r = client.post("/predict", json=payload_from_dev(50))
    assert r.status_code == 200
    assert r.json()["model_level"] in ("elevated", "high")


def test_invalid_product_type_rejected(client):
    bad = payload_from_dev(0) | {"product_type": "X"}
    assert client.post("/predict", json=bad).status_code == 422


def test_unknown_field_rejected(client):
    bad = payload_from_dev(0) | {"surprise": 1}
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_field_rejected(client):
    bad = payload_from_dev(0)
    del bad["torque_nm"]
    assert client.post("/predict", json=bad).status_code == 422


def test_out_of_bounds_value_rejected(client):
    bad = payload_from_dev(0) | {"torque_nm": 1000}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_ok(client):
    dev_ids = pd.read_csv("data/dev.csv")["row_id"].tolist()[:3]
    body = {"readings": [payload_from_dev(i) for i in dev_ids]}
    r = client.post("/predict/batch", json=body)
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_batch_too_large(client):
    body = {"readings": [payload_from_dev(0)] * 101}
    assert client.post("/predict/batch", json=body).status_code == 413


def test_batch_empty_rejected(client):
    assert client.post("/predict/batch", json={"readings": []}).status_code == 422