from tests.helpers import LOWPOWER, NORMAL


def test_predict_saves_prediction_and_returns_ids(client):
    r = client.post("/predict", json=NORMAL)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["prediction_id"], int)
    saved = client.get(f"/predictions/{body['prediction_id']}")
    assert saved.status_code == 200
    assert saved.json()["risk_score"] == body["risk_score"]
    assert saved.json()["warning_level"] == body["warning_level"]
    assert saved.json()["created_at"].endswith(("Z", "+00:00"))


def test_normal_reading_creates_no_alert(client):
    body = client.post("/predict", json=NORMAL).json()
    assert body["warning_level"] == "normal"
    assert body["alert_id"] is None and body["alert_created"] is False
    assert client.get("/alerts").json()["total"] == 0


def test_failure_creates_alert(client):
    body = client.post("/predict", json=LOWPOWER).json()
    assert body["alert_created"] is True and body["alert_id"] is not None
    alerts = client.get("/alerts").json()
    assert alerts["total"] == 1
    item = alerts["items"][0]
    assert item["status"] == "open"
    assert item["level"] == body["warning_level"]
    assert item["prediction_id"] == body["prediction_id"]
    assert item["model_version"] == body["model_version"]


def test_duplicate_reading_does_not_duplicate_alert(client):
    first = client.post("/predict", json=LOWPOWER).json()
    second = client.post("/predict", json=LOWPOWER).json()
    assert second["alert_created"] is False
    assert second["alert_id"] == first["alert_id"]
    assert client.get("/alerts").json()["total"] == 1
    assert client.get("/predictions").json()["total"] == 2  # history keeps both


def test_batch_dedups_within_batch(client):
    r = client.post("/predict/batch", json={"readings": [LOWPOWER, LOWPOWER, NORMAL]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert res[0]["alert_created"] is True
    assert res[1]["alert_created"] is False
    assert res[1]["alert_id"] == res[0]["alert_id"]
    assert res[2]["alert_id"] is None
    assert client.get("/alerts").json()["total"] == 1
    assert client.get("/predictions").json()["total"] == 3


def test_alert_lifecycle(client):
    alert_id = client.post("/predict", json=LOWPOWER).json()["alert_id"]
    ack = client.patch(f"/alerts/{alert_id}/status", json={"status": "acknowledged"})
    assert ack.status_code == 200
    assert ack.json()["status"] == "acknowledged"
    assert ack.json()["acknowledged_at"] is not None
    done = client.patch(f"/alerts/{alert_id}/status", json={"status": "resolved"})
    assert done.status_code == 200
    assert done.json()["resolved_at"] is not None
    again = client.patch(f"/alerts/{alert_id}/status", json={"status": "acknowledged"})
    assert again.status_code == 409


def test_resolved_alert_allows_new_alert(client):
    first = client.post("/predict", json=LOWPOWER).json()
    client.patch(f"/alerts/{first['alert_id']}/status", json={"status": "resolved"})
    second = client.post("/predict", json=LOWPOWER).json()
    assert second["alert_created"] is True
    assert second["alert_id"] != first["alert_id"]
    assert client.get("/alerts").json()["total"] == 2


def test_alert_filters(client):
    body = client.post("/predict", json=LOWPOWER).json()
    client.patch(f"/alerts/{body['alert_id']}/status", json={"status": "acknowledged"})
    assert client.get("/alerts?status=open").json()["total"] == 0
    assert client.get("/alerts?status=acknowledged").json()["total"] == 1
    assert client.get(f"/alerts?level={body['warning_level']}").json()["total"] == 1


def test_invalid_values_rejected(client):
    alert_id = client.post("/predict", json=LOWPOWER).json()["alert_id"]
    assert client.patch(f"/alerts/{alert_id}/status", json={"status": "open"}).status_code == 422
    assert client.get("/alerts?status=bogus").status_code == 422
    assert client.get("/predictions?warning_level=bogus").status_code == 422


def test_not_found(client):
    assert client.get("/alerts/9999").status_code == 404
    assert client.get("/predictions/9999").status_code == 404
    assert client.patch("/alerts/9999/status", json={"status": "resolved"}).status_code == 404


def test_pagination(client):
    readings = [NORMAL | {"tool_wear_min": w} for w in (100, 101, 102)]
    client.post("/predict/batch", json={"readings": readings})
    page = client.get("/predictions?limit=2").json()
    assert page["total"] == 3 and len(page["items"]) == 2
    assert len(client.get("/predictions?limit=2&offset=2").json()["items"]) == 1
    assert client.get("/predictions?limit=0").status_code == 422
    assert client.get("/predictions?limit=1000").status_code == 422


def test_predictions_filter_by_warning_level(client):
    high = client.post("/predict", json=LOWPOWER).json()["warning_level"]
    client.post("/predict", json=NORMAL)
    assert client.get(f"/predictions?warning_level={high}").json()["total"] == 1
    assert client.get("/predictions?warning_level=normal").json()["total"] == 1