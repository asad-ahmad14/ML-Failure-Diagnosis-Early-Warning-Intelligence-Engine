import json
from pathlib import Path

import pytest

from app.services.monitoring import feature_drift, psi, psi_label
from tests.helpers import NORMAL

ARTIFACTS = Path("artifacts")


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.1] * 10, [0.1] * 10) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_large_when_everything_moves_to_one_bin():
    assert psi([0.1] * 10, [0.0] * 9 + [1.0]) > 0.25


def test_psi_label_boundaries():
    assert psi_label(0.0999) == "stable"
    assert psi_label(0.1) == "moderate_shift"
    assert psi_label(0.2499) == "moderate_shift"
    assert psi_label(0.25) == "significant_shift"


REF = {"bin_edges": [1.0, 2.0, 3.0], "bin_expected": [0.25] * 4, "mean": 2.0}


def test_feature_drift_stable_when_values_match_reference():
    result = feature_drift([0.5, 1.5, 2.5, 3.5] * 10, REF)
    assert result["status"] == "stable"
    assert result["psi"] == pytest.approx(0.0, abs=1e-6)


def test_feature_drift_detects_shift():
    result = feature_drift([10.0] * 40, REF)
    assert result["status"] == "significant_shift"


def test_model_info_matches_saved_files(client):
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    saved_test = json.load(open(ARTIFACTS / "test_metrics.json"))
    saved_meta = json.load(open(ARTIFACTS / "model_metadata.json"))
    model = body["failure_risk_model"]
    assert model["held_out_test_metrics"]["pr_auc"] == saved_test["pr_auc"]
    assert model["thresholds"] == saved_meta["thresholds"]
    assert model["version"] == saved_meta["model_version"]
    assert len(model["dev_cross_validation"]) == 4
    assert body["limitations"]


def test_monitoring_report_empty(client):
    r = client.get("/monitoring/report")
    assert r.status_code == 200
    body = r.json()
    assert body["n_predictions"] == 0
    assert body["drift"]["status"] == "insufficient_data"
    assert body["alerts_by_status"] == {"open": 0, "acknowledged": 0, "resolved": 0}


def test_monitoring_report_needs_enough_samples_for_drift(client):
    client.post("/predict", json=NORMAL)
    body = client.get("/monitoring/report").json()
    assert body["n_predictions"] == 1
    assert body["drift"]["status"] == "insufficient_data"
    assert body["warning_level_counts"]["normal"] == 1


def test_monitoring_report_detects_drift(client):
    drifted = NORMAL | {"tool_wear_min": 250}
    r = client.post("/predict/batch", json={"readings": [drifted] * 40})
    assert r.status_code == 200
    body = client.get("/monitoring/report").json()
    assert body["n_predictions"] == 40
    drift = body["drift"]
    assert drift["status"] == "computed"
    assert drift["features"]["Tool wear"]["status"] == "significant_shift"
    assert drift["overall"] == "significant_shift"


def test_monitoring_window_validation(client):
    assert client.get("/monitoring/report?window=0").status_code == 422
    assert client.get("/monitoring/report?window=100000").status_code == 422