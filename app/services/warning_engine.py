"""Deterministic warning logic. Pure functions: no I/O, easy to test."""

LEVELS = ("normal", "watch", "elevated", "high")


def level_from_score(score: float, thresholds: dict) -> str:
    if score >= thresholds["high"]:
        return "high"
    if score >= thresholds["elevated"]:
        return "elevated"
    if score >= thresholds["watch"]:
        return "watch"
    return "normal"


def build_message(level: str, model_level: str, unusual: bool) -> str:
    if level == "normal":
        return ("No warning. The model risk score is below the Watch cutoff and the "
                "reading is not statistically unusual.")
    if level == "watch" and model_level == "normal":
        return ("This reading is statistically unusual compared with the training data, "
                "but the failure-risk model does not flag it. Unusual does not mean failing.")
    if level == "watch":
        return ("Watch: the model assigns a low-to-moderate risk score. Many readings at "
                "this level are not failures. Keep an eye on it.")
    if level == "elevated":
        return ("Elevated: the model assigns an elevated failure-risk score. This is a "
                "screening signal, not confirmation of mechanical failure.")
    return ("High priority: the model assigns a high failure-risk score. This is a "
            "screening signal, not confirmation of mechanical failure.")


def evaluate_warning(risk_score: float, thresholds: dict,
                     anomaly_score: float, anomaly_threshold: float) -> dict:
    model_level = level_from_score(risk_score, thresholds)
    unusual = anomaly_score >= anomaly_threshold
    level = model_level
    sources = []
    if model_level != "normal":
        sources.append("model")
    if unusual:
        sources.append("anomaly")
        if model_level == "normal":
            level = "watch"  # anomaly alone is capped at Watch
    return {
        "level": level,
        "model_level": model_level,
        "unusual": unusual,
        "sources": sources,
        "message": build_message(level, model_level, unusual),
    }