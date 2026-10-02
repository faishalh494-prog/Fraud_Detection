"""SyndicAI analyst console backed by the FastAPI investigation service."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SyndicAI | Investigation Desk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_request(path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    base_url = os.environ.get("SYNDICAI_API_URL", "http://127.0.0.1:8000").rstrip("/")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
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
        raise RuntimeError(f"Investigation API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach the FastAPI service at {base_url}. Start the API before opening this desk."
        ) from error


st.markdown(
    """
    <style>
      :root { --ink: #142638; --muted: #4e606e; --line: #cbd7de; --accent: #087e8b; }
      .stApp { background: #f5f7f8; color: var(--ink); }
      [data-testid="stSidebar"] { background: #102735; }
      [data-testid="stSidebar"] * { color: #edf4f6 !important; }
      .desk-kicker { color: #087e8b; font-size: .75rem; font-weight: 750; letter-spacing: .16em; }
      .desk-title { color: #142638; font-size: 1.9rem; font-weight: 760; margin: .15rem 0; }
      .desk-subtitle { color: var(--muted); margin-bottom: .65rem; }
      .desk-workflows { color: #304454; margin: .2rem 0 1rem; font-size: .92rem; }
      .risk-card { background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--accent);
                   border-radius: 6px; padding: .9rem 1rem; min-height: 104px; }
      .risk-label { color: #405565; font-size: .76rem; font-weight: 650;
                    text-transform: uppercase; letter-spacing: .06em; }
      .risk-value { color: #142638; font-size: 1.55rem; font-weight: 720; margin-top: .25rem; }
      [data-testid="stCaptionContainer"] { color: var(--muted); }
      [data-testid="stWidgetLabel"] p { color: #263d4f; font-weight: 600; }
      div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line);
                                    border-radius: 6px; padding: .75rem .9rem; }
      div[data-testid="stMetric"] label { color: #405565 !important; }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #142638; }
      div[data-testid="stDataFrame"] { border: 1px solid var(--line); }
      div[data-testid="stExpander"] { border-color: var(--line); border-radius: 6px; }
      div[data-testid="stTextInput"] input,
      div[data-testid="stNumberInput"] input,
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
    '<div class="desk-subtitle">Evidence-led review queue · risk scores guide attention, not conclusions</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="desk-workflows">Workflows: '
    '<a href="#historical-investigation">Historical investigation</a> · '
    '<a href="#new-transaction-review">New transaction review</a></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Case controls")
    selected_model = st.selectbox("Risk model", ["B", "C", "A"], format_func=lambda x: f"Model {x}")
    st.caption(
        {
            "A": "Transaction signals",
            "B": "Transaction + prior behaviour",
            "C": "Transaction + behaviour + prior network signals",
        }[selected_model]
    )
    st.divider()
    st.markdown("### Model evidence")
    try:
        model_report = api_request("/models")
        selected_report = model_report["models"][selected_model]
        st.write(f"Test PR-AUC · **{selected_report['test']['pr_auc']:.4f}**")
        st.write(f"Validation F1 · **{selected_report['validation']['f1']:.4f}**")
        st.write(f"Test recall · **{selected_report['test']['recall']:.1%}**")
        st.write(f"Test alerts · **{selected_report['test']['alerts']:,}**")
        st.caption("Threshold selected on validation data; final test metrics are held out.")
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

try:
    alert_rows = api_request(f"/alerts?model={selected_model}&limit=250")
except RuntimeError as error:
    st.error(str(error))
    st.stop()

st.subheader("Historical investigation", anchor="historical-investigation")
queue_col, case_col = st.columns([0.92, 1.5], gap="large")
with queue_col:
    st.subheader("Review queue")
    st.caption("Highest model scores from the held-out test period. Select a row to inspect.")
    if alert_rows:
        queue = pd.DataFrame(alert_rows)
        queue["case"] = queue.apply(
            lambda row: (
                f"#{int(row.row_index):,} · {row.transaction_type} · "
                f"{row.risk_score:.1f}/100"
            ),
            axis=1,
        )
        choice = st.selectbox("Transaction", queue["case"].tolist(), label_visibility="collapsed")
        selected_index = int(queue.loc[queue["case"] == choice, "row_index"].iloc[0])
        st.dataframe(
            queue[["row_index", "step", "transaction_type", "amount", "risk_score", "status"]]
            .rename(
                columns={
                    "row_index": "Row",
                    "step": "Step",
                    "transaction_type": "Type",
                    "amount": "Amount",
                    "risk_score": "Score",
                    "status": "Status",
                }
            ),
            hide_index=True,
            width="stretch",
            height=360,
        )
    else:
        st.info("No model-flagged transactions were found in the test-period queue.")
        selected_index = 0

    manual_index = st.number_input(
        "Inspect reference row",
        min_value=0,
        value=int(selected_index),
        step=1,
        help="Zero-based row from the validated processed PaySim reference file.",
    )

with case_col:
    st.subheader("Transaction investigation")
    try:
        case = api_request(f"/transactions/{int(manual_index)}?model={selected_model}")
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    transaction = case["transaction"]
    risk = case["risk"]
    if case["evaluation_period"] != "test":
        st.warning(case["score_context"])
    header_left, header_right = st.columns([1.5, 1])
    with header_left:
        st.markdown(f"**{transaction['transaction_type']}** · Step {transaction['step']}")
        st.text(f"Sender: {transaction['sender']}  →  Receiver: {transaction['receiver']}")
    with header_right:
        st.markdown(
            f'<div class="risk-card"><div class="risk-label">{risk["level"]}</div>'
            f'<div class="risk-value">{risk["score"]:.1f}<span style="font-size:.9rem"> / 100</span></div>'
            f'<div class="risk-label">Review threshold {risk["review_threshold"]:.1f}</div></div>',
            unsafe_allow_html=True,
        )

    amount_col, decision_col, model_col = st.columns(3)
    amount_col.metric("Transaction amount", f"{transaction['amount']:,.2f}")
    decision_col.metric("Review flag", "Flagged" if risk["flagged_for_review"] else "Below threshold")
    model_col.metric("Model", f"Model {case['model']}")
    st.caption(risk["score_kind"] + " · No score establishes that an account or transaction is fraudulent.")

    with st.expander("WHY THIS WAS FLAGGED", expanded=True):
        st.caption(
            f"{case['explanation']['method']}. Contributions show how the listed features moved "
            "this model's raw score; positive values increase it."
        )
        for reason in case["explanation"]["reasons"]:
            st.markdown(
                f"- **{reason['label']}** — `{reason['value']}` "
                f"({reason['direction']} score; contribution {reason['contribution']:+.4f})"
            )
        st.caption(case["explanation"]["caveat"])

    with st.expander("Behaviour before this transaction"):
        behaviour = case["behaviour"]
        if behaviour:
            st.dataframe(
                pd.DataFrame(
                    [{"Signal": key.replace("_", " ").title(), "Observed value": value} for key, value in behaviour.items()]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("Model A does not use behavioural history.")

    with st.expander("Network context · prior 24 steps"):
        graph = case["network_context"]
        st.caption(
            f"NetworkX directed ego context · {graph['observed_edge_count']} observed prior edges · "
            f"repeated sender → receiver relationship: {'yes' if graph['prior_relationship_seen'] else 'no'}"
        )
        network_col_1, network_col_2 = st.columns(2)
        network_col_1.markdown("**Sender's prior counterparties**")
        network_col_2.markdown("**Receiver's prior counterparties**")
        network_col_1.dataframe(
            pd.DataFrame(graph["sender"]["prior_receivers"]),
            hide_index=True,
            width="stretch",
        )
        network_col_2.dataframe(
            pd.DataFrame(graph["receiver"]["prior_senders"]),
            hide_index=True,
            width="stretch",
        )
        if case["network_features"]:
            st.json(case["network_features"])

    with st.expander("Investigation notes and status", expanded=True):
        current = case["investigation"]
        status = st.selectbox(
            "Status",
            ["Open", "Investigating", "Escalated", "Closed"],
            index=["Open", "Investigating", "Escalated", "Closed"].index(current["status"]),
            key=f"status_{manual_index}",
        )
        note = st.text_area("Analyst note", value=current["note"], key=f"note_{manual_index}")
        if st.button("Save investigation", type="primary"):
            try:
                saved = api_request(
                    f"/investigations/{int(manual_index)}",
                    method="PUT",
                    payload={"status": status, "note": note},
                )
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.success(f"Investigation saved · {saved['status']}")

st.divider()
st.subheader("New transaction review", anchor="new-transaction-review")
st.caption(
    "Submit a new event for API scoring. Its score prioritizes human review; "
    "it is not a fraud verdict. Successfully scored events enter online history."
)

try:
    transaction_limits = api_request("/transaction_limits")
    reference_max_step = int(transaction_limits["reference_max_step"])
except (RuntimeError, KeyError, TypeError, ValueError) as error:
    st.error(f"Cannot load the reference step limit: {error}")
else:
    with st.form("new_transaction_review"):
        transaction_col_1, transaction_col_2 = st.columns(2)
        with transaction_col_1:
            new_step = st.number_input(
                "Step",
                min_value=reference_max_step + 1,
                value=reference_max_step + 1,
                step=1,
                help=(
                    "Must be greater than the immutable V1 reference maximum "
                    f"step ({reference_max_step})."
                ),
            )
            new_type = st.selectbox(
                "Transaction type",
                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
            )
            new_amount = st.number_input(
                "Amount",
                min_value=0.0,
                value=0.0,
                step=100.0,
                format="%.2f",
            )
        with transaction_col_2:
            new_sender = st.text_input("Sender account")
            new_receiver = st.text_input("Receiver account")
            new_event_id = st.text_input(
                "Event ID (optional)",
                help="A repeated ID is rejected to prevent duplicate history updates.",
            )
            new_model = st.selectbox(
                "Scoring model",
                ["B", "A"],
                format_func=lambda model: f"Model {model}",
            )
        submitted = st.form_submit_button("Score new transaction", type="primary")

    if submitted:
        if not new_sender.strip() or not new_receiver.strip():
            st.error("Enter both a sender and receiver account.")
        else:
            payload: dict[str, Any] = {
                "step": int(new_step),
                "type": new_type,
                "amount": float(new_amount),
                "nameOrig": new_sender.strip(),
                "nameDest": new_receiver.strip(),
                "model": new_model,
            }
            if new_event_id.strip():
                payload["event_id"] = new_event_id.strip()
            try:
                new_case = api_request(
                    "/score_transaction",
                    method="POST",
                    payload=payload,
                )
            except RuntimeError as error:
                message = str(error)
                if "HTTP 409" in message:
                    st.error(f"Duplicate event ID: {message}")
                elif "HTTP 422" in message:
                    st.error(f"Transaction was not accepted: {message}")
                else:
                    st.error(message)
            else:
                risk = new_case["risk"]
                st.success(
                    "Transaction scored and added to online history. "
                    "A later transaction at a greater step can use it as prior behaviour."
                )
                st.markdown(
                    f"**{new_type}** · Step {int(new_step)}"
                )
                st.text(
                    f"Sender: {new_sender.strip()}  →  Receiver: {new_receiver.strip()}"
                )
                metric_cols = st.columns(4)
                metric_cols[0].metric("Model score", f"{risk['score']:.2f}/100")
                metric_cols[1].metric("Risk level", risk["risk_level"])
                metric_cols[2].metric(
                    "Review threshold",
                    f"{risk['review_threshold']:.2f}/100",
                )
                metric_cols[3].metric(
                    "Review status",
                    "Flagged" if risk["flagged_for_review"] else "Below threshold",
                )
                st.caption(
                    f"{risk['score_kind']}. A score is not a fraud verdict. "
                    f"{new_case['history_rule']}"
                )
                with st.expander("WHY THIS WAS FLAGGED", expanded=True):
                    st.caption(new_case["explanation"]["summary"])
                    for reason in new_case["explanation"]["reasons"]:
                        st.markdown(
                            f"- **{reason['label']}** — `{reason['value']}` "
                            f"({reason['direction']} score; contribution "
                            f"{reason['contribution']:+.4f})"
                        )
                    st.caption(new_case["explanation"]["caveat"])
                with st.expander("Behavioural evidence used"):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"Signal": key.replace("_", " ").title(), "Value": value}
                                for key, value in new_case["behavioural_evidence"].items()
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                st.caption(
                    "Online history updated · "
                    f"event ID: {new_event_id.strip() or 'not supplied'} · "
                    f"history updated: {'yes' if new_case['history_updated'] else 'no'}"
                )
