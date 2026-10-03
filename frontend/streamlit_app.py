"""SyndicAI investigator workspace backed by the FastAPI service."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import pandas as pd
import streamlit as st

WORKFLOWS = (
    "OVERVIEW",
    "LIVE MONITOR",
    "ALERTS",
    "INVESTIGATE",
    "NEW TRANSACTION",
    "MODEL / EVIDENCE",
)
MODEL_NAMES = {
    "A": "Transaction signals",
    "B": "Transaction + prior behaviour",
    "C": "Transaction + prior behaviour + prior network signals",
}
STATUSES = ["Open", "Investigating", "Escalated", "Closed"]

st.set_page_config(
    page_title="SyndicAI | Investigation Desk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_request(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> Any:
    base_url = os.environ.get("SYNDICAI_API_URL", "http://127.0.0.1:8000").rstrip("/")
    api_key = os.environ.get("SYNDICAI_API_KEY", "")
    if len(api_key) < 32:
        raise RuntimeError(
            "Set the same 32-character-or-longer SYNDICAI_API_KEY for the API and dashboard."
        )
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(message).get("detail", message)
        except (json.JSONDecodeError, AttributeError):
            detail = message
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach the FastAPI service at {base_url}. Start the API before opening this desk."
        ) from error


def load_required(path: str, description: str) -> Any:
    try:
        return api_request(path)
    except RuntimeError as error:
        st.error(f"{description}: {error}")
        st.stop()


def model_selector() -> str:
    with st.sidebar:
        st.markdown("### Historical risk model")
        selected = st.selectbox(
            "Model",
            ["B", "C", "A"],
            format_func=lambda model: f"Model {model}",
            key="historical_model",
            label_visibility="collapsed",
        )
        st.caption(MODEL_NAMES[selected])
        st.caption("New transaction review selects its model independently.")
    return selected


def get_alerts(model: str, limit: int = 250) -> list[dict[str, Any]]:
    return load_required(
        f"/alerts?model={model}&limit={limit}",
        "Could not load the review queue",
    )


def alert_label(row: dict[str, Any]) -> str:
    return (
        f"Row {int(row['row_index']):,} · {row['transaction_type']} · "
        f"Step {int(row['step'])} · {float(row['risk_score']):.1f}/100"
    )


def evidence_frame(reasons: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Evidence": reason["label"],
                "Observed value": reason["value"],
                "Effect": reason["direction"].title(),
                "Contribution": f"{float(reason['contribution']):+.4f}",
            }
            for reason in reasons
        ]
    )


def render_risk_assessment(risk: dict[str, Any]) -> None:
    score_col, priority_col, threshold_col, probability_col, status_col = st.columns(5)
    score_col.metric("Risk score", f"{risk['score']:.2f}/100")
    priority_col.metric("Review priority", risk["review_priority"])
    threshold_col.metric("Review threshold", f"{risk['review_threshold']:.2f}/100")
    probability_col.metric(
        "Calibrated probability",
        "Not available" if risk["calibrated_probability"] is None else f"{risk['calibrated_probability']:.1%}",
    )
    status_col.metric(
        "Review status",
        "Flagged" if risk["flagged_for_review"] else "Below threshold",
    )
    st.caption(
        f"{risk['score_kind']} · Not a calibrated probability or fraud verdict."
    )


def render_evidence_strength(evidence: dict[str, Any]) -> None:
    st.markdown("#### Evidence strength · behavioural history")
    st.metric("History coverage", evidence["status"])
    if evidence["sender_prior_transactions"] is not None:
        sender_col, receiver_col = st.columns(2)
        sender_col.metric(
            "Sender prior transactions",
            evidence["sender_prior_transactions"],
        )
        receiver_col.metric(
            "Receiver prior transactions",
            evidence["receiver_prior_transactions"],
        )
    st.caption(
        f"{evidence['interpretation']} Established history means at least "
        f"{evidence['established_history_minimum']} prior events for both accounts."
    )


def render_explanation(
    explanation: dict[str, Any],
    *,
    flagged: bool,
) -> None:
    heading = "WHY THIS WAS FLAGGED" if flagged else "WHY THIS SCORE"
    st.markdown(f"#### {heading}")
    st.caption(
        f"{explanation['method']}. Positive contributions increase the model score; "
        "negative contributions decrease it."
    )
    reasons = evidence_frame(explanation["reasons"])
    if not reasons.empty:
        st.dataframe(reasons, hide_index=True, width="stretch")
    st.caption(explanation["caveat"])


def render_behaviour(behaviour: dict[str, Any], *, empty_message: str) -> None:
    if not behaviour:
        st.info(empty_message)
        return
    st.dataframe(
        pd.DataFrame(
            [
                {"Behavioural evidence": key.replace("_", " ").title(), "Value": value}
                for key, value in behaviour.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def render_network(graph: dict[str, Any], network_features: dict[str, Any]) -> None:
    st.caption(
        f"Prior 24-step NetworkX context · {graph['observed_edge_count']} observed edges · "
        f"prior sender → receiver relationship: "
        f"{'seen' if graph['prior_relationship_seen'] else 'not seen'}"
    )
    sender_col, receiver_col = st.columns(2)
    with sender_col:
        st.markdown("**Sender’s prior receivers**")
        sender_rows = graph["sender"]["prior_receivers"]
        if sender_rows:
            st.dataframe(pd.DataFrame(sender_rows), hide_index=True, width="stretch")
        else:
            st.caption("No prior receivers in this context window.")
    with receiver_col:
        st.markdown("**Receiver’s prior senders**")
        receiver_rows = graph["receiver"]["prior_senders"]
        if receiver_rows:
            st.dataframe(pd.DataFrame(receiver_rows), hide_index=True, width="stretch")
        else:
            st.caption("No prior senders in this context window.")
    if network_features:
        st.markdown("**Network model inputs**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Network signal": key.replace("_", " ").title(), "Value": value}
                    for key, value in network_features.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def open_investigation() -> None:
    st.session_state["selected_row_index"] = int(st.session_state["alert_row_choice"])
    st.session_state["dashboard_workflow"] = "INVESTIGATE"


def page_overview(model: str) -> None:
    report = load_required("/models", "Could not load model evidence")
    health = load_required("/health", "Could not load system status")
    model_report = report["models"][model]
    alert_preview = get_alerts(model, limit=250)
    high_priority_count = sum(
        row["review_priority"] == "High review priority" for row in alert_preview
    )
    total_alerts = int(model_report["test"]["alerts"])
    high_label = (
        f"{high_priority_count} of top {len(alert_preview)}"
        if alert_preview
        else "0 in preview"
    )

    st.markdown("### Situational awareness")
    status_col, total_col, high_col, model_col = st.columns(4)
    status_col.metric(
        "System status",
        "Ready" if health["status"] == "ready" else "Artifacts missing",
    )
    total_col.metric("Test-period review alerts", f"{total_alerts:,}")
    high_col.metric("High priority in queue preview", high_label)
    model_col.metric("Current model", f"Model {model}")
    st.caption(
        "High-priority count covers the top alert preview only; the total alert count "
        "comes from the fixed test evaluation."
    )

    st.markdown("### What SyndicAI does")
    st.write(
        "SyndicAI ranks transactions for analyst review using transaction signals "
        "and, in Model B, behaviour observed at earlier steps. Use the evidence and "
        "context below to decide what to investigate next; a score is not a fraud finding."
    )

    st.markdown("### Validated Model B evidence")
    baseline = report["models"]["B"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Test PR-AUC", f"{baseline['test']['pr_auc']:.4f}")
    metric_cols[1].metric("Test precision", f"{baseline['test']['precision']:.1%}")
    metric_cols[2].metric("Test recall", f"{baseline['test']['recall']:.1%}")
    metric_cols[3].metric("Test F1", f"{baseline['test']['f1']:.4f}")
    st.caption(
        "Fixed chronological PaySim holdout · validation-selected threshold "
        f"{baseline['validation']['threshold']:.4f} · "
        f"{baseline['test']['alerts']:,} test alerts. These are measured results, "
        "not production performance claims."
    )

    if alert_preview:
        st.markdown("### Needs attention")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Transaction": f"{row['transaction_type']} · row {row['row_index']:,}",
                        "Step": row["step"],
                        "Amount": f"{row['amount']:,.2f}",
                        "Model score": f"{row['risk_score']:.1f}/100",
                        "Review priority": row["review_priority"],
                        "Status": row["status"],
                    }
                    for row in alert_preview[:5]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.button(
            "Open Alerts",
            on_click=lambda: st.session_state.update({"dashboard_workflow": "ALERTS"}),
        )


@st.fragment(run_every="2s")
def page_live_monitor() -> None:
    """Refresh the durable event feed while this workflow is open."""
    status = load_required("/live/status", "Could not load live-stream status")
    response = load_required(
        "/live/events?limit=50",
        "Could not load recent live events",
    )
    events = response["events"]
    latest = events[0] if events else None

    st.markdown("### Live event monitoring")
    st.caption(
        "Incoming events are scored sequentially through Model B and committed "
        "to SQLite only after feature generation, inference, and explanation succeed. "
        "This local stream prototype is not a production streaming service."
    )
    event_col, alert_col, state_col, latency_col = st.columns(4)
    event_col.metric("Scored live events", f"{status['event_count']:,}")
    alert_col.metric("Flagged for review", f"{status['flagged_event_count']:,}")
    state_col.metric(
        "Latest event",
        "Done" if latest else "Waiting",
    )
    latency_col.metric(
        "Latest processing work",
        f"{latest['timings']['processing_ms']:.1f} ms" if latest else "—",
    )
    st.caption(
        "Automatic refresh every 2 seconds · "
        f"latest step: {status['latest_step'] if status['latest_step'] is not None else 'none'} · "
        f"latest receipt: {status['latest_processed_at'] or 'no events yet'}"
    )

    if not events:
        st.info(
            "No scored live events yet. Start the API, then run "
            "`python -m demo.live_stream --interval 1.0` in another terminal."
        )
        return

    st.markdown("#### Latest events")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Event": event["event_key"],
                    "Step": event["transaction"]["step"],
                    "Transaction": event["transaction"]["type"],
                    "Amount": event["transaction"]["amount"],
                    "Risk score": event["risk"]["score"],
                    "Review priority": event["risk"]["review_priority"],
                    "Flagged": event["risk"]["flagged_for_review"],
                    "Behavioural history": event["evidence_strength"]["status"],
                    "Processing": event["processing_state"],
                }
                for event in events
            ]
        ).style.format(
            {"Amount": "{:,.2f}", "Risk score": "{:.2f}/100"}
        ),
        hide_index=True,
        width="stretch",
        height=320,
    )

    flagged_events = [
        event for event in events if event["risk"]["flagged_for_review"]
    ]
    review_options = flagged_events or events
    event_lookup = {event["event_key"]: event for event in review_options}
    selected_key = st.selectbox(
        "Open a live event for investigation",
        list(event_lookup),
        format_func=lambda key: (
            f"{event_lookup[key]['transaction']['type']} · "
            f"step {event_lookup[key]['transaction']['step']} · "
            f"score {event_lookup[key]['risk']['score']:.2f}/100"
        ),
        key="live_event_choice",
    )
    selected = event_lookup[selected_key]
    st.markdown("#### Live investigation")
    transaction = selected["transaction"]
    st.write(
        f"**{transaction['type']}** · step **{transaction['step']}** · "
        f"amount **{transaction['amount']:,.2f}**"
    )
    st.text(f"Sender: {transaction['sender']}  →  Receiver: {transaction['receiver']}")
    render_risk_assessment(selected["risk"])
    render_evidence_strength(selected["evidence_strength"])
    render_explanation(
        selected["explanation"],
        flagged=selected["risk"]["flagged_for_review"],
    )
    st.markdown("#### Behaviour before this event")
    render_behaviour(
        selected["behavioural_evidence"],
        empty_message="This event did not use behavioural features.",
    )
    st.caption(
        f"Event {selected['event_key']} · "
        f"feature {selected['timings']['feature_ms']:.2f} ms · "
        f"inference {selected['timings']['inference_ms']:.2f} ms · "
        f"explanation {selected['timings']['explanation_ms']:.2f} ms · "
        f"state write {selected['timings']['state_update_ms']:.2f} ms"
    )


def page_alerts(model: str) -> None:
    report = load_required("/models", "Could not load model evidence")
    rows = get_alerts(model)
    metrics = report["models"][model]["test"]
    st.markdown("### Prioritize investigations")
    st.caption(
        f"Top {len(rows)} scored alerts from the fixed test period · "
        f"{metrics['alerts']:,} total alerts at the validation-selected threshold."
    )
    if not rows:
        st.info("No alerts are available for this model.")
        return

    table = pd.DataFrame(
        [
            {
                "Transaction": f"{row['transaction_type']} · row {row['row_index']:,}",
                "Step": row["step"],
                "Amount": row["amount"],
                "Risk score": row["risk_score"],
                "Review priority": row["review_priority"],
                "Status": row["status"],
            }
            for row in rows
        ]
    )
    st.dataframe(
        table.style.format({"Amount": "{:,.2f}", "Risk score": "{:.1f}/100"}),
        hide_index=True,
        width="stretch",
        height=480,
    )
    st.markdown("#### Investigate an alert")
    row_options = [int(row["row_index"]) for row in rows]
    row_lookup = {int(row["row_index"]): row for row in rows}
    default_row = int(st.session_state.get("selected_row_index", row_options[0]))
    if default_row not in row_lookup:
        default_row = row_options[0]
    st.selectbox(
        "Alert",
        row_options,
        index=row_options.index(default_row),
        format_func=lambda index: alert_label(row_lookup[index]),
        key="alert_row_choice",
    )
    st.button(
        "Investigate selected alert",
        type="primary",
        on_click=open_investigation,
    )


def page_investigate(model: str) -> None:
    report = load_required("/models", "Could not load model evidence")
    rows = get_alerts(model)
    total_rows = sum(report["dataset"]["split_rows"].values())
    row_options = [int(row["row_index"]) for row in rows]
    row_lookup = {int(row["row_index"]): row for row in rows}

    st.markdown("### Select a historical transaction")
    source = st.radio(
        "Transaction source",
        ["Alert queue", "Reference row"],
        horizontal=True,
        index=0 if row_options else 1,
    )
    selected_default = int(st.session_state.get("selected_row_index", 0))
    if source == "Alert queue" and row_options:
        selected_default = (
            selected_default if selected_default in row_lookup else row_options[0]
        )
        selected_row = st.selectbox(
            "Alert queue transaction",
            row_options,
            index=row_options.index(selected_default),
            format_func=lambda index: alert_label(row_lookup[index]),
            key="investigation_alert_choice",
        )
    else:
        selected_row = st.number_input(
            "Or enter reference row",
            min_value=0,
            max_value=max(total_rows - 1, 0),
            value=min(selected_default, max(total_rows - 1, 0)),
            step=1,
            key="investigation_manual_row",
            help="Zero-based row in the immutable validated reference dataset.",
        )
    selected_row = int(selected_row)
    st.session_state["selected_row_index"] = selected_row

    try:
        case = api_request(f"/transactions/{selected_row}?model={model}")
    except RuntimeError as error:
        st.error(f"Could not load this transaction: {error}")
        return

    transaction = case["transaction"]
    risk = case["risk"]
    st.divider()
    st.markdown("### 1 · Transaction summary")
    summary_left, summary_amount, summary_period = st.columns([2, 1, 1])
    summary_left.markdown(
        f"**{transaction['transaction_type']}** · Step **{transaction['step']}**"
    )
    summary_left.text(
        f"Sender: {transaction['sender']}  →  Receiver: {transaction['receiver']}"
    )
    summary_amount.metric("Amount", f"{transaction['amount']:,.2f}")
    summary_period.metric("Evaluation period", case["evaluation_period"].title())
    if case["evaluation_period"] != "test":
        st.warning(case["score_context"])

    st.markdown("### 2 · Risk assessment")
    render_risk_assessment(
        risk
    )
    render_evidence_strength(case["evidence_strength"])
    st.markdown("### 3 · Why this was flagged")
    render_explanation(case["explanation"], flagged=risk["flagged_for_review"])

    st.markdown("### 4 · Behaviour before transaction")
    render_behaviour(
        case["behaviour"],
        empty_message="This model does not use behavioural history.",
    )

    st.markdown("### 5 · Network context")
    render_network(case["network_context"], case["network_features"])

    st.markdown("### 6 · Investigation action")
    current = case["investigation"]
    with st.form(f"investigation_{selected_row}"):
        status = st.selectbox(
            "Investigation status",
            STATUSES,
            index=STATUSES.index(current["status"]),
        )
        note = st.text_area("Analyst notes", value=current["note"], max_chars=2000)
        save = st.form_submit_button("Save investigation", type="primary")
    if save:
        try:
            saved = api_request(
                f"/investigations/{selected_row}",
                method="PUT",
                payload={"status": status, "note": note},
            )
        except RuntimeError as error:
            st.error(f"Could not save investigation: {error}")
        else:
            st.success(f"Investigation updated · {saved['status']}")


def render_new_transaction_result(result: dict[str, Any]) -> None:
    transaction = result["transaction"]
    st.divider()
    st.markdown("### 2 · Risk assessment")
    st.markdown(
        f"**{transaction['type']}** · Step **{transaction['step']}**"
    )
    st.text(f"Sender: {transaction['sender']}  →  Receiver: {transaction['receiver']}")
    render_risk_assessment(result["risk"])
    render_evidence_strength(result["evidence_strength"])
    st.markdown("### 3 · Why")
    render_explanation(
        result["explanation"],
        flagged=result["risk"]["flagged_for_review"],
    )
    st.markdown("### 4 · Behavioural evidence")
    render_behaviour(
        result["behavioural_evidence"],
        empty_message="Model A uses transaction signals only; no behavioural features were used.",
    )
    st.markdown("### 5 · History update")
    if result["history_updated"]:
        st.success(
            "Event added to online history. A later event at a greater step can use it."
        )
    st.caption(result["history_rule"])
    st.caption(
        f"Event ID: {result.get('event_id') or 'not supplied'} · "
        f"{result['risk']['score_kind']} · This is not a fraud verdict."
    )


def page_new_transaction() -> None:
    limits = load_required(
        "/transaction_limits",
        "Could not load the new-transaction step boundary",
    )
    reference_max_step = int(limits["reference_max_step"])
    operating_points_response = load_required(
        "/operating_points",
        "Could not load validation-selected alert operating points",
    )
    operating_points = operating_points_response["operating_points"]
    point_by_id = {point["id"]: point for point in operating_points}
    policy_ids = ["max_f1", *(point["id"] for point in operating_points if point["id"] != "max_f1")]
    policy_labels = {point["id"]: point["label"] for point in operating_points}
    st.markdown("### Enter a transaction")
    st.caption(
        "Score a genuinely new event. Steps must be greater than the immutable "
        f"reference maximum ({reference_max_step}); same-step events do not see each other."
    )
    model = st.selectbox(
        "Model",
        ["B", "A"],
        format_func=lambda name: f"Model {name}",
        index=0,
        key="new_transaction_model",
    )
    operating_point = st.selectbox(
        "Alert operating point",
        policy_ids if model == "B" else ["max_f1"],
        format_func=lambda point_id: policy_labels[point_id],
        key="new_transaction_operating_point",
        help=(
            "Every alternative is selected from validation scores. "
            "The existing maximum-F1 threshold remains the default."
        ),
    )
    point = point_by_id[operating_point]
    validation_metrics = point["validation"]
    test_metrics = point["test"]
    st.caption(
        f"Selected review threshold: {point['threshold'] * 100:.2f}/100 · "
        f"validation alerts: {validation_metrics['alerts']:,} "
        f"({validation_metrics['alert_burden_pct']:.3f}%) · "
        f"historical test alerts: {test_metrics['alerts']:,}, including "
        f"{test_metrics['false_positives']:,} false positives "
        f"({test_metrics['alert_burden_pct']:.3f}% burden). "
        "Test workload is historical evidence, not a future-volume guarantee."
    )
    with st.form("new_transaction_review"):
        transaction_col, accounts_col = st.columns(2)
        with transaction_col:
            step = st.number_input(
                "Step",
                min_value=reference_max_step + 1,
                value=reference_max_step + 1,
                step=1,
            )
            transaction_type = st.selectbox(
                "Transaction type",
                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
            )
            amount = st.number_input(
                "Amount",
                min_value=0.0,
                value=0.0,
                step=100.0,
                format="%.2f",
            )
        with accounts_col:
            sender = st.text_input("Sender account")
            receiver = st.text_input("Receiver account")
            event_id = st.text_input(
                "Event ID (optional)",
                help="Repeated IDs are rejected with HTTP 409.",
            )
        submitted = st.form_submit_button("Score transaction", type="primary")

    if submitted:
        if not sender.strip() or not receiver.strip():
            st.error("Enter both a sender account and a receiver account.")
        elif int(step) <= reference_max_step:
            st.error(
                f"Step must be greater than the reference maximum ({reference_max_step})."
            )
        else:
            payload: dict[str, Any] = {
                "step": int(step),
                "type": transaction_type,
                "amount": float(amount),
                "nameOrig": sender.strip(),
                "nameDest": receiver.strip(),
                "model": model,
                "operating_point": operating_point,
            }
            if event_id.strip():
                payload["event_id"] = event_id.strip()
            try:
                result = api_request(
                    "/score_transaction",
                    method="POST",
                    payload=payload,
                )
            except RuntimeError as error:
                message = str(error)
                if message.startswith("HTTP 409"):
                    st.error(f"Duplicate event ID: {message}")
                elif message.startswith("HTTP 422"):
                    st.error(f"Transaction not accepted: {message}")
                else:
                    st.error(message)
            else:
                result["transaction"] = {
                    "type": transaction_type,
                    "step": int(step),
                    "amount": float(amount),
                    "sender": sender.strip(),
                    "receiver": receiver.strip(),
                }
                result["event_id"] = payload.get("event_id")
                st.session_state["latest_new_transaction_result"] = result
                st.success("Transaction scored and added to online history.")

    latest = st.session_state.get("latest_new_transaction_result")
    if latest:
        st.markdown("### 1 · Scoring result")
        render_new_transaction_result(latest)


def page_model_evidence() -> None:
    report = load_required("/models", "Could not load model evidence")
    st.markdown("### Validated Model A / B / C comparison")
    rows = []
    for name in ("A", "B", "C"):
        metrics = report["models"][name]
        validation = metrics["validation"]
        test = metrics["test"]
        rows.append(
            {
                "Model": f"Model {name}",
                "Signals": MODEL_NAMES[name],
                "Validation threshold": validation["threshold"],
                "Test precision": test["precision"],
                "Test recall": test["recall"],
                "Test F1": test["f1"],
                "Test PR-AUC": test["pr_auc"],
                "Test alerts": test["alerts"],
                "Alert burden": test["alert_burden_pct"] / 100,
            }
        )
    comparison = pd.DataFrame(rows)
    st.dataframe(
        comparison.style.format(
            {
                "Validation threshold": "{:.6f}",
                "Test precision": "{:.1%}",
                "Test recall": "{:.1%}",
                "Test F1": "{:.4f}",
                "Test PR-AUC": "{:.4f}",
                "Test alerts": "{:,.0f}",
                "Alert burden": "{:.4%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown("### Why Model B is the current default")
    st.write(
        "On the recorded fixed PaySim holdout, Model B improved test precision, "
        "F1, and PR-AUC over transaction-only Model A while producing fewer alerts. "
        "Model C tied Model B in the recorded run, and its network features had "
        "zero gain importance. Model B is therefore the simpler evidence-backed "
        "default; network context remains useful for investigation, not as a "
        "demonstrated predictive improvement."
    )
    st.caption(
        "Thresholds are selected on validation data. Test metrics are held out, "
        "based on synthetic PaySim data, and are not production performance claims."
    )
    st.markdown("### Additional investigation signals")
    st.write(
        "NetworkX displays prior counterparties and relationship context for "
        "historical investigations. The normal-behaviour anomaly detector is a "
        "rejected offline experiment; it is not part of production scoring."
    )
    st.caption(
        "The alert-budget evaluation shows that test alert burden differs from "
        "validation targets. No alternative Model B threshold is deployed."
    )


st.markdown(
    """
    <style>
      :root { --ink: #142638; --muted: #4e606e; --line: #cbd7de; --accent: #087e8b; }
      .stApp { background: #f5f7f8; color: var(--ink); }
      [data-testid="stSidebar"] { background: #102735; }
      [data-testid="stSidebar"] * { color: #edf4f6 !important; }
      .desk-kicker { color: #087e8b; font-size: .74rem; font-weight: 750; letter-spacing: .15em; }
      .desk-title { color: #142638; font-size: 1.9rem; font-weight: 760; margin: .15rem 0; }
      .desk-subtitle { color: var(--muted); margin-bottom: .6rem; }
      [data-testid="stCaptionContainer"] { color: var(--muted); }
      [data-testid="stWidgetLabel"] p { color: #263d4f; font-weight: 600; }
      div[data-testid="stRadio"] label p { color: #263d4f !important; }
      div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line);
                                    border-radius: 5px; padding: .75rem .9rem; }
      div[data-testid="stMetric"] label { color: #405565 !important; }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #142638; }
      div[data-testid="stDataFrame"] { border: 1px solid var(--line); }
      div[data-testid="stExpander"] { border-color: var(--line); border-radius: 5px; }
      div[data-testid="stTextInput"] input,
      div[data-testid="stNumberInput"] input,
      div[data-testid="stTextArea"] textarea,
      div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #fff; border-color: #9aabb7; color: #142638;
      }
      .stButton button[kind="primary"],
      .stFormSubmitButton button[kind="primary"] {
        background: #087e8b; border-color: #087e8b; color: #fff;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="desk-kicker">SYNDICAI / FRAUD INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="desk-title">Investigation desk</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="desk-subtitle">Transaction review · evidence for analyst decisions, '
    'not fraud verdicts</div>',
    unsafe_allow_html=True,
)
workflow = st.radio(
    "Investigator workflow",
    WORKFLOWS,
    horizontal=True,
    key="dashboard_workflow",
    label_visibility="collapsed",
)

if workflow in {"ALERTS", "INVESTIGATE"}:
    selected_model = model_selector()
else:
    selected_model = str(st.session_state.get("historical_model", "B"))

if workflow == "OVERVIEW":
    page_overview(selected_model)
elif workflow == "LIVE MONITOR":
    page_live_monitor()
elif workflow == "ALERTS":
    page_alerts(selected_model)
elif workflow == "INVESTIGATE":
    page_investigate(selected_model)
elif workflow == "NEW TRANSACTION":
    page_new_transaction()
else:
    page_model_evidence()
