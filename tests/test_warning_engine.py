import pytest

from app.services.warning_engine import evaluate_warning, level_from_score

THR = {"watch": 0.1, "elevated": 0.4, "high": 0.5}


@pytest.mark.parametrize(
    "score,expected",
    [(0.0, "normal"), (0.0999, "normal"), (0.1, "watch"), (0.3999, "watch"),
     (0.4, "elevated"), (0.4999, "elevated"), (0.5, "high"), (1.0, "high")],
)
def test_levels_at_boundaries(score, expected):
    assert level_from_score(score, THR) == expected


def test_anomaly_alone_raises_normal_to_watch():
    w = evaluate_warning(0.01, THR, anomaly_score=0.7, anomaly_threshold=0.6)
    assert w["level"] == "watch"
    assert w["model_level"] == "normal"
    assert w["sources"] == ["anomaly"]


def test_anomaly_never_escalates_above_model_level():
    w = evaluate_warning(0.45, THR, anomaly_score=0.9, anomaly_threshold=0.6)
    assert w["level"] == "elevated"
    assert w["sources"] == ["model", "anomaly"]


def test_anomaly_threshold_is_inclusive():
    assert evaluate_warning(0.0, THR, 0.6, 0.6)["unusual"] is True
    assert evaluate_warning(0.0, THR, 0.5999, 0.6)["unusual"] is False


def test_normal_has_no_sources():
    w = evaluate_warning(0.0, THR, 0.1, 0.6)
    assert w["level"] == "normal" and w["sources"] == []