"""Streamlit dashboard. Every number shown comes from the running API."""
import os

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("MFE_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 60.0

st.set_page_config(page_title="ML Failure Diagnosis Engine", layout="wide")


def call(method: str, path: str, **kwargs):
    """Return (data, error_message). Never raises into the page."""
    try:
        r = httpx.request(method, f"{API}{path}", timeout=TIMEOUT, **kwargs)
    except httpx.HTTPError as exc:
        return None, f"Cannot reach the API at {API} ({type(exc).__name__})."
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"API returned {r.status_code}: {detail}"
    return r.json(), None


def show_warning(level: str, text: str) -> None:
    {"normal": st.success, "watch": st.info, "elevated": st.warning, "high": st.error}[level](text)


def bar(series: pd.Series) -> None:
    """Bar chart that always starts at zero (the default auto axis can dip below it)."""
    data = series.rename_axis("label").reset_index(name="value")
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=None, title=None),
            y=alt.Y("value:Q", scale=alt.Scale(domainMin=0), title=None),
        )
    )
    st.altair_chart(chart)


st.title("ML Failure Diagnosis & Early-Warning Engine")
st.caption(
    "Screening signals from a model trained on the synthetic AI4I 2020 dataset. "
    "Not confirmation of mechanical failure."
)

health, health_err = call("GET", "/health")
with st.sidebar:
    st.header("Service")
    if health_err:
        st.error(health_err)
    else:
        st.write(f"Status: **{health['status']}**")
        st.write(f"Database: {health['database']}")
        st.write(f"Risk model: {health['model_version']}")
        st.write(f"Anomaly model: {health['anomaly_model_version']}")
    st.caption(f"API: {API}")
if health_err:
    st.stop()

tab_check, tab_hist, tab_alerts, tab_model = st.tabs(
    ["Score a reading", "Prediction history", "Alerts", "Model and monitoring"]
)

# ---------------------------------------------------------------- score a reading
with tab_check:
    st.subheader("Score a sensor reading")
    st.caption("The starting values are only an example input.")
    with st.form("reading"):
        c1, c2, c3 = st.columns(3)
        product_type = c1.selectbox("Product type", ["L", "M", "H"], index=1)
        air = c1.number_input("Air temperature (K)", min_value=250.0, max_value=350.0, value=300.0, step=0.1)
        proc = c2.number_input("Process temperature (K)", min_value=250.0, max_value=400.0, value=310.0, step=0.1)
        rpm = c2.number_input("Rotational speed (rpm)", min_value=1.0, max_value=10000.0, value=1500.0, step=10.0)
        torque = c3.number_input("Torque (Nm)", min_value=0.0, max_value=500.0, value=40.0, step=0.5)
        wear = c3.number_input("Tool wear (min)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
        submitted = st.form_submit_button("Score reading")

    if submitted:
        payload = {
            "product_type": product_type,
            "air_temperature_k": air,
            "process_temperature_k": proc,
            "rotational_speed_rpm": rpm,
            "torque_nm": torque,
            "tool_wear_min": wear,
        }
        data, err = call("POST", "/predict", json=payload)
        if err:
            st.error(err)
        else:
            st.session_state["last"] = data

    result = st.session_state.get("last")
    if result:
        show_warning(
            result["warning_level"],
            f"Warning level: {result['warning_level'].upper()}. {result['message']}",
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Model risk score", f"{result['risk_score']:.3f}")
        m2.metric("Anomaly score", f"{result['anomaly_score']:.3f}")
        m3.metric("Statistically unusual", "yes" if result["is_statistically_unusual"] else "no")
        st.write("Warning sources:", ", ".join(result["warning_sources"]) or "none")

        st.markdown("**Model evidence** (top features by contribution to the risk score)")
        ev = pd.DataFrame(result["evidence"])
        # Round numbers for display; str keeps the column one type, which Arrow requires.
        ev["value"] = ev["value"].apply(
            lambda v: f"{v:.5g}" if isinstance(v, (int, float)) else str(v)
        )
        ev = ev.rename(columns={"contribution": "contribution (score units)"})
        st.dataframe(ev)
        for note in result["data_quality_notes"]:
            st.warning(note)
        if result["alert_id"]:
            state = "created" if result["alert_created"] else "already active, reused"
            st.write(f"Alert #{result['alert_id']} ({state}). Prediction #{result['prediction_id']}.")
        else:
            st.write(f"No alert. Prediction #{result['prediction_id']}.")
        st.caption(result["disclaimer"])
        with st.expander("Thresholds used for this result"):
            st.json(result["thresholds"])

# ---------------------------------------------------------------- history
with tab_hist:
    st.subheader("Recent predictions")
    c1, c2 = st.columns(2)
    level = c1.selectbox("Warning level", ["all", "normal", "watch", "elevated", "high"], key="hist_level")
    limit = c2.number_input("Rows", min_value=10, max_value=200, value=50, step=10, key="hist_limit")
    params = {"limit": int(limit)}
    if level != "all":
        params["warning_level"] = level
    page, err = call("GET", "/predictions", params=params)
    if err:
        st.error(err)
    elif page["total"] == 0:
        st.info("No predictions stored yet. Score a reading or run scripts/simulate_traffic.py.")
    else:
        df = pd.DataFrame(page["items"])
        df["warning_sources"] = df["warning_sources"].apply(", ".join)
        cols = ["id", "created_at", "product_type", "risk_score", "anomaly_score",
                "warning_level", "warning_sources", "model_version"]
        st.caption(f"Showing {len(df)} of {page['total']} stored predictions, newest first.")
        st.dataframe(df[cols])

# ---------------------------------------------------------------- alerts
with tab_alerts:
    st.subheader("Alerts")
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)
    c1, c2 = st.columns(2)
    status = c1.selectbox("Status", ["all", "open", "acknowledged", "resolved"], key="alert_status")
    alevel = c2.selectbox("Level", ["all", "watch", "elevated", "high"], key="alert_level")
    params = {"limit": 100}
    if status != "all":
        params["status"] = status
    if alevel != "all":
        params["level"] = alevel
    page, err = call("GET", "/alerts", params=params)
    if err:
        st.error(err)
    elif page["total"] == 0:
        st.info("No alerts match these filters.")
    else:
        df = pd.DataFrame(page["items"])
        df["sources"] = df["sources"].apply(", ".join)
        st.caption(f"Showing {len(df)} of {page['total']} alerts, newest first.")
        st.dataframe(df[["id", "prediction_id", "created_at", "level", "status", "sources", "model_version"]])

        st.markdown("**Update an alert**")
        alert_id = st.selectbox("Alert id", df["id"].tolist(), key="alert_pick")
        current = df[df["id"] == alert_id].iloc[0]["status"]
        actions = {"open": ["acknowledged", "resolved"], "acknowledged": ["resolved"], "resolved": []}[current]
        st.write(f"Current status: **{current}**")
        if not actions:
            st.info("This alert is resolved; no further changes are possible.")
        for col, target in zip(st.columns(len(actions) or 1), actions):
            if col.button(f"Mark {target}", key=f"btn_{alert_id}_{target}"):
                _, perr = call("PATCH", f"/alerts/{int(alert_id)}/status", json={"status": target})
                if perr:
                    st.error(perr)
                else:
                    st.session_state["flash"] = f"Alert #{alert_id} is now {target}."
                    st.rerun()

# ---------------------------------------------------------------- model and monitoring
with tab_model:
    info, err = call("GET", "/model")
    if err:
        st.error(err)
    else:
        model = info["failure_risk_model"]
        st.subheader("Held-out test evaluation")
        st.caption(
            "Read from " + ", ".join(info["metric_sources"])
            + ". These files are written by the training and evaluation scripts; "
            "none of these numbers are typed into the dashboard."
        )
        tm = model["held_out_test_metrics"]
        if tm:
            a, b = st.columns(2)
            a.metric("Test PR-AUC", f"{tm['pr_auc']:.3f}")
            b.metric("Test ROC-AUC", f"{tm['roc_auc']:.3f}")
            rows = []
            for name, m in tm["levels"].items():
                rows.append({
                    "level": name,
                    "cutoff": m["threshold"],
                    "precision": m["precision"],
                    "precision 95% interval": f"{m['precision_ci95'][0]:.2f} to {m['precision_ci95'][1]:.2f}",
                    "recall": m["recall"],
                    "recall 95% interval": f"{m['recall_ci95'][0]:.2f} to {m['recall_ci95'][1]:.2f}",
                    "false alarms": m["fp"],
                    "missed": m["fn"],
                    "flagged per 1000": m["flagged_per_1000"],
                })
            st.dataframe(pd.DataFrame(rows))
            st.caption("The test set has about 68 failures, so these figures carry wide uncertainty.")
        else:
            st.info("test_metrics.json not found. Run python -m ml.evaluate_test.")

        st.subheader("Dev-set cross-validation")
        if model["dev_cross_validation"]:
            st.dataframe(pd.DataFrame(model["dev_cross_validation"]).round(4))
        st.subheader("Global feature importance")
        imp = model["global_importance"]
        if imp:
            bar(pd.Series(imp["importance"]).sort_values())
            st.caption("Mean |SHAP| on a dev sample. Shows how the model uses inputs, not physical causes.")
        with st.expander("Threshold rule and limitations"):
            st.write(model["threshold_rule"])
            for item in info["limitations"]:
                st.write("- " + item)

    st.subheader("Live monitoring")
    report, err = call("GET", "/monitoring/report")
    if err:
        st.error(err)
    elif report["n_predictions"] == 0:
        st.info("No predictions stored yet.")
    else:
        a, b, c = st.columns(3)
        a.metric("Predictions in window", report["n_predictions"])
        b.metric("Share flagged unusual", f"{report['share_statistically_unusual']:.1%}")
        c.metric("Share with range notes", f"{report['share_with_range_notes']:.1%}")
        st.write("Alerts by status:", report["alerts_by_status"])
        if report["warning_level_counts"]:
            bar(pd.Series(report["warning_level_counts"]))
        drift = report["drift"]
        if drift["status"] != "computed":
            st.info(f"Drift needs at least {drift['min_samples']} stored readings (have {drift['n']}).")
        else:
            st.write(f"Overall input drift: **{drift['overall']}** (max PSI {drift['max_psi']})")
            st.dataframe(pd.DataFrame([{"feature": k, **v} for k, v in drift["features"].items()]))
            st.dataframe(pd.DataFrame([{"Type": k, **v} for k, v in drift["type_share"].items()]))
        for note in report["notes"]:
            st.caption(note)