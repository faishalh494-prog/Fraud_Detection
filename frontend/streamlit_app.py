"""SyndicAI investigator workspace backed by the FastAPI service."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any
from urllib.parse import quote

import pandas as pd
import streamlit as st

WORKFLOWS = (
    "OVERVIEW",
    "LIVE MONITOR",
    "ALERTS",
    "INVESTIGATIONS",
    "HISTORICAL INVESTIGATION",
    "CUSTOMERS",
    "NETWORK INTELLIGENCE",
    "MODEL / EVIDENCE",
    "TRANSACTION HISTORY",
    "SYSTEM",
    "NEW TRANSACTION",
)
WORKFLOW_LABELS = {
    "OVERVIEW": "Overview",
    "LIVE MONITOR": "Live Monitor",
    "ALERTS": "Alerts",
    "INVESTIGATIONS": "Investigations",
    "HISTORICAL INVESTIGATION": "Historical Investigation",
    "CUSTOMERS": "Customers",
    "NETWORK INTELLIGENCE": "Network Intelligence",
    "MODEL / EVIDENCE": "Model Evidence",
    "TRANSACTION HISTORY": "Transaction History",
    "SYSTEM": "System",
    "NEW TRANSACTION": "New Transaction",
}
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


def render_risk_assessment(
    risk: dict[str, Any],
    *,
    compact: bool = False,
) -> None:
    if compact:
        first_row = st.columns(2)
        first_row[0].metric("Risk score", f"{risk['score']:.2f}/100")
        first_row[1].metric("Review priority", risk["review_priority"])
        second_row = st.columns(2)
        second_row[0].metric(
            "Review threshold",
            f"{risk['review_threshold']:.2f}/100",
        )
        second_row[1].metric(
            "Review status",
            "Flagged" if risk["flagged_for_review"] else "Below threshold",
        )
    else:
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
        f"{risk['score_kind']} · Calibrated probability: Not available · "
        "not a fraud verdict."
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
    st.session_state["dashboard_workflow"] = "HISTORICAL INVESTIGATION"


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


def local_timestamp(value: str | None) -> str:
    if not value:
        return "Time unavailable"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "Time unavailable"
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone().strftime("%H:%M:%S")


def event_title(event: dict[str, Any]) -> str:
    transaction = event["transaction"]
    risk = event["risk"]
    flag = "●" if risk["flagged_for_review"] else "○"
    return (
        f"{flag} {local_timestamp(event.get('processed_at'))} · "
        f"{transaction['type']} · {transaction['amount']:,.2f} · "
        f"{risk['score']:.1f}/100"
    )


def event_payload_path(event_key: str, suffix: str = "") -> str:
    return f"/live/events/{quote(event_key, safe='')}{suffix}"


def save_live_investigation(
    event_key: str,
    *,
    status: str,
    note: str,
) -> bool:
    try:
        saved = api_request(
            event_payload_path(event_key, "/investigation"),
            method="PUT",
            payload={"status": status, "note": note},
        )
    except RuntimeError as error:
        st.error(f"Could not update investigation: {error}")
        return False
    st.session_state["live_investigation_saved"] = saved
    return True


def render_contribution_bars(reasons: list[dict[str, Any]]) -> None:
    st.markdown("#### WHY THIS EVENT NEEDS REVIEW")
    st.caption("MODEL EVIDENCE — NOT PROOF · TreeSHAP contribution to this model score")
    if not reasons:
        st.info("No feature contributions were returned for this event.")
        return
    maximum = max(abs(float(reason["contribution"])) for reason in reasons) or 1.0
    for reason in reasons[:5]:
        direction = str(reason["direction"]).lower()
        symbol = "+" if float(reason["contribution"]) > 0 else "−"
        st.markdown(
            f"**{reason['label']}**  ·  {reason['value']}  ·  "
            f"{symbol}{abs(float(reason['contribution'])):.4f}"
        )
        st.progress(
            min(abs(float(reason["contribution"])) / maximum, 1.0),
            text=direction,
        )


def behavior_timeline(event: dict[str, Any]) -> pd.DataFrame:
    transaction = event["transaction"]
    rows: list[dict[str, Any]] = []
    prior_events = event.get("related_activity", [])
    if prior_events:
        prior_amounts = [
            {
                "Step": int(item["step"]),
                "Amount": float(item["amount"]),
                "Activity": str(item.get("type") or "Online transaction"),
            }
            for item in reversed(prior_events[-12:])
        ]
        rows.extend(prior_amounts)
    rows.append(
        {
            "Step": int(transaction["step"]),
            "Amount": float(transaction["amount"]),
            "Activity": f"Current · {transaction['type']}",
        }
    )
    return pd.DataFrame(rows)


def render_live_case_actions(event: dict[str, Any]) -> None:
    event_key = str(event["event_key"])
    investigation = event.get("investigation", {})
    current_status = investigation.get("status", "Open")
    current_note = investigation.get("note", "")
    st.markdown("#### INVESTIGATOR ACTION")
    status_col, action_col = st.columns([1, 1])
    status_col.caption(
        f"Current status: **{current_status}**"
        + (
            f" · updated {local_timestamp(investigation['updated_at'])}"
            if investigation.get("updated_at")
            else ""
        )
    )
    if action_col.button(
        "Open investigation",
        key=f"open_live_{event_key}",
        disabled=current_status == "Investigating",
        width="stretch",
    ):
        if save_live_investigation(
            event_key,
            status="Investigating",
            note=current_note,
        ):
            st.rerun(scope="fragment")

    action_cols = st.columns(2)
    if action_cols[0].button(
        "Escalate",
        key=f"escalate_live_{event_key}",
        width="stretch",
    ):
        if save_live_investigation(
            event_key,
            status="Escalated",
            note=current_note,
        ):
            st.rerun(scope="fragment")
    if action_cols[1].button(
        "Close",
        key=f"close_live_{event_key}",
        width="stretch",
    ):
        if save_live_investigation(
            event_key,
            status="Closed",
            note=current_note,
        ):
            st.rerun(scope="fragment")

    with st.form(f"live_note_{event_key}"):
        note = st.text_area(
            "Analyst note",
            value=current_note,
            max_chars=2000,
            key=f"live_note_value_{event_key}",
        )
        status = st.selectbox(
            "Investigation status",
            STATUSES,
            index=STATUSES.index(current_status),
            key=f"live_status_value_{event_key}",
        )
        add_note = st.form_submit_button("Save status and note", type="primary")
    if add_note and save_live_investigation(
        event_key,
        status=status,
        note=note,
    ):
        st.rerun(scope="fragment")


@st.fragment(run_every="2s")
def page_live_monitor() -> None:
    """Render the live river and selected event from authenticated API data."""
    st.markdown("### Live event monitoring")
    st.caption("Automatic refresh every 2 seconds while this view is open.")
    status = load_required("/live/status", "Could not load live-stream status")
    response = load_required(
        "/live/events?limit=50",
        "Could not load recent live events",
    )
    events = response["events"]

    if not events:
        st.info(
            "No scored events are available yet. Start the API and run "
            "`python -m demo.live_stream --interval 1.0`."
        )
        return

    selected_key = st.session_state.get("selected_live_event_key")
    available_keys = {str(event["event_key"]) for event in events}
    if selected_key not in available_keys:
        selected_key = str(events[0]["event_key"])
        st.session_state["selected_live_event_key"] = selected_key

    river_col, detail_col = st.columns([0.9, 1.2], gap="large")
    with river_col:
        st.markdown("### Transaction river")
        st.caption(
            f"Most recent {len(events)} scored events · newest first · "
            f"{status['event_count']:,} processed in persistent local history"
        )
        filter_mode = st.radio(
            "Event filter",
            ["All events", "Flagged only"],
            horizontal=True,
            key="live_event_filter",
            label_visibility="collapsed",
        )
        visible = (
            [event for event in events if event["risk"]["flagged_for_review"]]
            if filter_mode == "Flagged only"
            else events
        )
        if not visible:
            st.info("No flagged events are present in the current feed window.")
        else:
            with st.container(height=610, border=True):
                for event in visible:
                    key = str(event["event_key"])
                    transaction = event["transaction"]
                    is_selected = key == selected_key
                    if st.button(
                        event_title(event),
                        key=f"river_event_{key}",
                        type="primary" if is_selected else "secondary",
                        width="stretch",
                    ):
                        st.session_state["selected_live_event_key"] = key
                        st.rerun(scope="fragment")
                    st.caption(
                        f"{key} · step {transaction['step']} · "
                        f"{transaction['sender']} → {transaction['receiver']} · "
                        f"{event['risk']['review_priority']} · "
                        f"{event.get('investigation', {}).get('status', 'Unknown')}"
                    )

    try:
        selected = api_request(event_payload_path(str(selected_key)))
    except RuntimeError as error:
        with detail_col:
            st.error(f"Could not load selected live event: {error}")
        return

    with detail_col:
        transaction = selected["transaction"]
        st.markdown("### Transaction investigation")
        st.caption(
            f"{selected['event_key']} · step {transaction['step']} · received "
            f"{local_timestamp(selected.get('processed_at'))} · "
            f"{selected['investigation']['status']}"
        )
        st.markdown(
            f"**{transaction['type']}** · **{transaction['amount']:,.2f}**"
        )
        st.caption(
            f"{transaction['sender']}  →  {transaction['receiver']}"
        )
        render_risk_assessment(selected["risk"], compact=True)
        render_contribution_bars(selected["explanation"]["reasons"])
        render_evidence_strength(selected["evidence_strength"])

        tab_behavior, tab_related, tab_network = st.tabs(
            ["Behaviour", "Related activity", "Network context"]
        )
        with tab_behavior:
            features = selected["behavioural_evidence"]
            metric_cols = st.columns(2)
            metric_cols[0].metric(
                "Sender prior events",
                features.get("sender_txn_count_before", "Not used"),
            )
            metric_cols[1].metric(
                "Receiver prior events",
                features.get("receiver_txn_count_before", "Not used"),
            )
            receiver_average = float(
                features.get("receiver_avg_amount_before", 0) or 0
            )
            current_amount = float(transaction["amount"])
            if receiver_average > 0:
                st.caption(
                    f"Current amount / receiver prior average: "
                    f"{current_amount / receiver_average:.2f}× "
                    f"({current_amount:,.2f} vs {receiver_average:,.2f})."
                )
            elif features:
                st.caption("No receiver prior average is available for comparison.")
            else:
                st.caption("Model A does not use behavioural features.")
            st.caption(
                f"Receiver activity in prior 24 steps: "
                f"{features.get('receiver_txn_count_last24_before', 'Not used')} · "
                "values are causal model features, not a fraud verdict."
            )
            st.markdown("**Recent online behavioural timeline**")
            st.caption(
                "Prior scored online events only, shown chronologically; "
                "reference-history counts are summarized above."
            )
            st.dataframe(
                behavior_timeline(selected),
                hide_index=True,
                width="stretch",
                height=220,
            )
        with tab_related:
            related = selected.get("related_activity", [])
            if related:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Time": local_timestamp(row.get("processed_at")),
                                "Step": row["step"],
                                "Type": row.get("type") or "Transaction",
                                "Amount": row["amount"],
                                "Sender → receiver": f"{row['sender']} → {row['receiver']}",
                                "Risk score": row.get("risk_score"),
                                "Case": row["investigation_status"],
                            }
                            for row in related
                        ]
                    ).style.format({"Amount": "{:,.2f}", "Risk score": "{:.2f}"}),
                    hide_index=True,
                    width="stretch",
                    height=240,
                )
            else:
                st.info("No earlier related online events are stored for these accounts.")
        with tab_network:
            st.caption(selected["network_context_scope"])
            context = selected["network_context"]
            st.caption(
                f"Immutable reference: prior {context['window_steps']} steps · "
                f"{context['observed_edge_count']} edges · "
                f"sender→receiver seen: "
                f"{'yes' if context['prior_relationship_seen'] else 'no'}."
            )
            for heading, rows in (
                ("Reference sender prior receivers", context["sender"]["prior_receivers"]),
                ("Reference receiver prior senders", context["receiver"]["prior_senders"]),
                (
                    "Online sender prior receivers",
                    selected["online_network_context"]["sender_prior_receivers"],
                ),
                (
                    "Online receiver prior senders",
                    selected["online_network_context"]["receiver_prior_senders"],
                ),
            ):
                st.markdown(f"**{heading}**")
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                else:
                    st.caption("No observed counterparties in this source/window.")

        st.caption(
            f"Feature {selected['timings']['feature_ms']:.1f} ms · "
            f"inference {selected['timings']['inference_ms']:.1f} ms · "
            f"explanation {selected['timings']['explanation_ms']:.1f} ms · "
            f"total {selected['timings']['processing_ms']:.1f} ms"
        )
        render_live_case_actions(selected)

    st.divider()
    st.markdown("### Recent alert activity")
    flagged = [
        event for event in events if event["risk"]["flagged_for_review"]
    ]
    if flagged:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Event": event["event_key"],
                        "Received": local_timestamp(event.get("processed_at")),
                        "Step": event["transaction"]["step"],
                        "Risk score": event["risk"]["score"],
                        "Review priority": event["risk"]["review_priority"],
                        "Investigation": event.get("investigation", {}).get(
                            "status", "Unknown"
                        ),
                    }
                    for event in flagged
                ]
            ).style.format({"Risk score": "{:.2f}/100"}),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No review alerts in the current live event window.")


@st.fragment(run_every="2s")
def system_bar() -> None:
    health = load_required("/health", "Could not read API health")
    status = load_required("/live/status", "Could not read live event status")
    latest_at = status.get("latest_processed_at")
    if latest_at:
        try:
            latest = datetime.fromisoformat(str(latest_at).replace("Z", "+00:00"))
            if latest.tzinfo is None:
                latest = latest.astimezone()
            age_seconds = max(
                0.0,
                (datetime.now().astimezone() - latest.astimezone()).total_seconds(),
            )
            flow_state = "Recent" if age_seconds <= 30 else "Idle"
        except ValueError:
            age_seconds = None
            flow_state = "Unknown"
    else:
        age_seconds = None
        flow_state = "Waiting"
    values = st.columns(4)
    values[0].metric(
        "API health",
        "Ready" if health["status"] == "ready" else "Unavailable",
    )
    values[1].metric("Recent event flow", flow_state)
    values[2].metric("Events processed", f"{status['event_count']:,}")
    values[3].metric(
        "Latest step",
        status["latest_step"] if status["latest_step"] is not None else "—",
    )
    latest_processing_ms = status.get("latest_processing_ms")
    st.caption(
        f"Active model: {status.get('active_model', 'Unknown')} · "
        "latest processing: "
        + (
            f"{latest_processing_ms:.1f} ms"
            if latest_processing_ms is not None
            else "not reported"
        )
        + " · "
        f"latest event: {local_timestamp(latest_at)}"
        + (
            f" ({age_seconds:.0f}s ago)"
            if age_seconds is not None
            else ""
        )
        + " · event recency is not a producer heartbeat."
    )


def live_feed(limit: int = 200) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    response = load_required(
        f"/live/events?limit={limit}",
        "Could not load live event history",
    )
    return response, response["events"]


def show_live_event(event_key: str) -> None:
    st.session_state["selected_live_event_key"] = event_key
    st.session_state["dashboard_workflow"] = "LIVE MONITOR"
    st.rerun()


def page_live_alerts() -> None:
    _, events = live_feed()
    flagged = [event for event in events if event["risk"]["flagged_for_review"]]
    st.markdown("### Live alerts")
    st.caption(
        f"{len(flagged)} flagged event(s) in the most recent {len(events)} persisted "
        "scores. The global count is shown in the system bar."
    )
    if not flagged:
        st.info("There are no flagged events in the available live event window.")
        return
    for event in flagged:
        transaction = event["transaction"]
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(
                f"**{transaction['type']}** · step {transaction['step']} · "
                f"{transaction['amount']:,.2f} · "
                f"risk {event['risk']['score']:.2f}/100"
            )
            left.caption(
                f"{event['event_key']} · {local_timestamp(event.get('processed_at'))} · "
                f"{transaction['sender']} → {transaction['receiver']} · "
                f"{event['investigation']['status']}"
            )
            right.button(
                "Investigate",
                key=f"inspect_alert_{event['event_key']}",
                width="stretch",
                on_click=show_live_event,
                args=(str(event["event_key"]),),
            )


def page_investigations() -> None:
    _, events = live_feed()
    cases = [
        event for event in events
        if event.get("investigation", {}).get("status") not in (None, "Open")
    ]
    st.markdown("### Investigations")
    st.caption(
        "Persisted live-event case states from SQLite. Historical/reference "
        "investigations remain available in the Historical investigation view."
    )
    if cases:
        for event in cases:
            transaction = event["transaction"]
            with st.container(border=True):
                left, right = st.columns([4, 1])
                left.markdown(
                    f"**{event.get('investigation', {}).get('status', 'Unknown')}** · "
                    f"{transaction['type']} · step {transaction['step']} · "
                    f"risk {event['risk']['score']:.2f}/100"
                )
                left.caption(
                    f"{event['event_key']} · {transaction['sender']} → "
                    f"{transaction['receiver']} · "
                    f"{event.get('investigation', {}).get('note') or 'No analyst note'}"
                )
                right.button(
                    "Open case",
                    key=f"open_case_{event['event_key']}",
                    width="stretch",
                    on_click=show_live_event,
                    args=(str(event["event_key"]),),
                )
    else:
        st.info("No live cases have been moved from Open.")
    with st.expander("Historical reference investigations"):
        model = st.selectbox(
            "Historical model",
            ["B", "C", "A"],
            format_func=lambda value: f"Model {value}",
            key="investigation_history_model",
        )
        historical = get_alerts(model)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Row": row["row_index"],
                        "Type": row["transaction_type"],
                        "Step": row["step"],
                        "Amount": row["amount"],
                        "Risk score": row["risk_score"],
                        "Review priority": row["review_priority"],
                        "Status": row["status"],
                    }
                    for row in historical
                ]
            ).style.format({"Amount": "{:,.2f}", "Risk score": "{:.1f}/100"}),
            hide_index=True,
            width="stretch",
        )
        if historical:
            row_index = st.selectbox(
                "Open historical alert",
                [int(row["row_index"]) for row in historical],
                format_func=lambda index: next(
                    alert_label(row) for row in historical if row["row_index"] == index
                ),
            )
            if st.button("Investigate historical transaction", type="primary"):
                st.session_state["selected_row_index"] = int(row_index)
                st.session_state["dashboard_workflow"] = "HISTORICAL INVESTIGATION"
                st.rerun()


def page_customers() -> None:
    _, events = live_feed()
    observed: dict[str, dict[str, Any]] = {}
    for event in events:
        transaction = event["transaction"]
        for role, account in (
            ("Sender", transaction["sender"]),
            ("Receiver", transaction["receiver"]),
        ):
            row = observed.setdefault(
                str(account),
                {"account": str(account), "transactions": 0, "roles": set(), "events": []},
            )
            row["transactions"] += 1
            row["roles"].add(role)
            row["events"].append(event)
    st.markdown("### Customers and accounts")
    st.caption(
        "Account activity observed in the most recent persisted live events only; "
        "this is not a complete customer profile or identity record."
    )
    if not observed:
        st.info("No live event account activity is available yet.")
        return
    account = st.selectbox("Observed account", sorted(observed))
    selected = observed[account]
    columns = st.columns(3)
    columns[0].metric("Events in feed window", selected["transactions"])
    columns[1].metric("Observed roles", " / ".join(sorted(selected["roles"])))
    columns[2].metric("Risk", "Review flagged" if any(
        event["risk"]["flagged_for_review"] for event in selected["events"]
    ) else "No flagged event in window")
    rows = []
    for event in selected["events"]:
        transaction = event["transaction"]
        rows.append(
            {
                "Event": event["event_key"],
                "Time": local_timestamp(event.get("processed_at")),
                "Step": transaction["step"],
                "Role": (
                    "Sender" if transaction["sender"] == account else "Receiver"
                ),
                "Counterparty": (
                    transaction["receiver"]
                    if transaction["sender"] == account
                    else transaction["sender"]
                ),
                "Type": transaction["type"],
                "Amount": transaction["amount"],
                "Risk score": event["risk"]["score"],
                "Investigation": event.get("investigation", {}).get(
                    "status", "Unknown"
                ),
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {"Amount": "{:,.2f}", "Risk score": "{:.2f}/100"}
        ),
        hide_index=True,
        width="stretch",
    )


def page_network_intelligence() -> None:
    _, events = live_feed()
    st.markdown("### Network intelligence")
    st.caption(
        "Live online counterparties and prior reference NetworkX context are "
        "reported separately. No relationship is inferred beyond observed transactions."
    )
    if not events:
        st.info("No live event is available to inspect for network context.")
        return
    event_key = st.selectbox(
        "Live event",
        [str(event["event_key"]) for event in events],
        format_func=lambda key: next(
            f"{event['transaction']['type']} · step {event['transaction']['step']} · {key}"
            for event in events if event["event_key"] == key
        ),
        key="network_event_choice",
    )
    try:
        details = api_request(event_payload_path(event_key))
    except RuntimeError as error:
        st.error(f"Could not load network context: {error}")
        return
    event = details["transaction"]
    st.write(
        f"Focus: **{event['sender']} → {event['receiver']}** · "
        f"step {event['step']}."
    )
    st.markdown("#### Immutable reference context")
    render_network(details["network_context"], {})
    st.markdown("#### Earlier online counterparties")
    online = details["online_network_context"]
    sender_col, receiver_col = st.columns(2)
    for column, title, rows in (
        (sender_col, "Sender's online prior receivers", online["sender_prior_receivers"]),
        (receiver_col, "Receiver's online prior senders", online["receiver_prior_senders"]),
    ):
        with column:
            st.markdown(f"**{title}**")
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            else:
                st.caption("No matching earlier scored online transactions.")


def page_transaction_history() -> None:
    response, events = live_feed()
    st.markdown("### Transaction history")
    st.caption(
        f"{response['event_count']:,} scored online events persisted; displaying "
        f"the {len(events)} most recent. Immutable reference rows are available "
        "through Historical investigation."
    )
    if not events:
        st.info("No online transaction history has been recorded.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Event": event["event_key"],
                    "Received": local_timestamp(event.get("processed_at")),
                    "Step": event["transaction"]["step"],
                    "Type": event["transaction"]["type"],
                    "Amount": event["transaction"]["amount"],
                    "Sender": event["transaction"]["sender"],
                    "Receiver": event["transaction"]["receiver"],
                    "Risk score": event["risk"]["score"],
                    "Priority": event["risk"]["review_priority"],
                    "Case status": event.get("investigation", {}).get(
                        "status", "Unknown"
                    ),
                }
                for event in events
            ]
        ).style.format({"Amount": "{:,.2f}", "Risk score": "{:.2f}/100"}),
        hide_index=True,
        width="stretch",
        height=540,
    )


def page_system() -> None:
    health = load_required("/health", "Could not read API health")
    status = load_required("/live/status", "Could not read live event status")
    st.markdown("### System")
    cols = st.columns(4)
    cols[0].metric("API", health["status"].replace("_", " ").title())
    cols[1].metric("API key configured", "Yes" if health["api_key_configured"] else "No")
    cols[2].metric("Reference maximum step", status["reference_max_step"])
    cols[3].metric("Latest online step", status["latest_step"] or "—")
    st.caption(
        "Event recency is used only to indicate recent flow; it is not a "
        "producer heartbeat or a guarantee that a producer is running."
    )
    st.json(
        {
            "live_status": status,
            "system_health": health,
            "live_feed_limit": 200,
            "dashboard_refresh_seconds": 2,
        }
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
      :root { --ink:#e3ebef; --muted:#9aabb4; --line:#2b3d47;
              --panel:#13232d; --panel-hi:#192c37; --accent:#54b8ad; }
      .stApp { background:#0b151c; color:var(--ink); }
      [data-testid="stHeader"] { background:#0b151c; }
      [data-testid="stSidebar"] { background:#0e1b23; border-right:1px solid var(--line); }
      section[data-testid="stSidebar"] {
        width:260px !important; min-width:260px !important;
      }
      [data-testid="stSidebar"] * { color:#dce6eb; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius:4px; padding:.24rem .45rem; margin:.06rem 0;
      }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background:#1a3038; border-left:3px solid var(--accent);
      }
      .block-container { max-width:1900px; padding-top:.9rem; padding-bottom:1.6rem; }
      .desk-kicker { color:var(--accent); font-size:.72rem; font-weight:750; letter-spacing:.14em; }
      .desk-title { color:#f0f5f7; font-size:1.65rem; font-weight:720; margin:.12rem 0; }
      .desk-subtitle { color:var(--muted); margin-bottom:.35rem; }
      h1,h2,h3,h4 { color:#eaf0f3; letter-spacing:-.02em; }
      [data-testid="stCaptionContainer"] { color:var(--muted); }
      [data-testid="stWidgetLabel"] p { color:#c3d0d6; font-weight:600; }
      div[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line);
                                    border-radius:4px; padding:.48rem .65rem; }
      div[data-testid="stMetric"] label { color:#a8b8c0 !important; font-size:.74rem; }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#edf4f6; font-size:1.35rem; }
      div[data-testid="stDataFrame"] { border:1px solid var(--line); }
      div[data-testid="stExpander"],div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color:var(--line); border-radius:4px;
      }
      div[data-testid="stTextInput"] input,
      div[data-testid="stNumberInput"] input,
      div[data-testid="stTextArea"] textarea,
      div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background:#10212a; border-color:#38505b; color:#e3ebef;
      }
      div[data-testid="stRadio"] label p { color:#c7d2d7 !important; }
      .stButton button { background:#172833; border-color:#30434d; color:#dce6eb; }
      .stButton button[kind="primary"],
      .stFormSubmitButton button[kind="primary"] {
        background:#216c69; border-color:#347e79; color:#f3fbfa;
      }
      .stButton button:hover { border-color:var(--accent); color:#fff; }
      div[data-testid="stTabs"] button[aria-selected="true"] {
        color:var(--accent); border-bottom-color:var(--accent);
      }
      div[data-testid="stProgress"] > div > div { background:var(--accent); }
      hr { border-color:var(--line); }
      @media (max-width: 1100px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="desk-kicker">SYNDICAI / FRAUD INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="desk-title">Live investigation workspace</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="desk-subtitle">Transaction monitoring · model evidence for investigator decisions, '
    'not fraud verdicts</div>',
    unsafe_allow_html=True,
)
system_bar()
st.divider()
with st.sidebar:
    st.markdown("### ◈ SyndicAI")
    st.caption("INVESTIGATION CONSOLE")
    workflow = st.radio(
        "Navigation",
        WORKFLOWS,
        format_func=lambda item: WORKFLOW_LABELS[item],
        key="dashboard_workflow",
        label_visibility="collapsed",
    )

if workflow == "OVERVIEW":
    selected_model = model_selector()
    page_overview(selected_model)
elif workflow == "LIVE MONITOR":
    page_live_monitor()
elif workflow == "ALERTS":
    page_live_alerts()
elif workflow == "INVESTIGATIONS":
    page_investigations()
elif workflow == "CUSTOMERS":
    page_customers()
elif workflow == "NETWORK INTELLIGENCE":
    page_network_intelligence()
elif workflow == "TRANSACTION HISTORY":
    page_transaction_history()
elif workflow == "SYSTEM":
    page_system()
elif workflow == "HISTORICAL INVESTIGATION":
    selected_model = model_selector()
    page_investigate(selected_model)
elif workflow == "NEW TRANSACTION":
    page_new_transaction()
elif workflow == "MODEL / EVIDENCE":
    page_model_evidence()
