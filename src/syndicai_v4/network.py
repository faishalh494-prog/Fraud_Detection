"""Leakage-safe temporal network features and investigator graph context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

NETWORK_FEATURES = [
    "network_sender_prior_in_degree",
    "network_sender_in_degree_last24",
    "network_receiver_prior_out_degree",
    "network_receiver_out_degree_last24",
]


def _entity_history(events: pd.DataFrame, entity: str) -> pd.DataFrame:
    events = events.copy()
    events[entity] = events[entity].astype(object)
    summary = (
        events.groupby([entity, "step"], sort=True, observed=True)
        .size()
        .rename("step_count")
        .reset_index()
        .sort_values([entity, "step"])
        .reset_index(drop=True)
    )
    summary["cumulative"] = summary.groupby(entity, sort=False, observed=True)["step_count"].cumsum()
    summary["before"] = summary["cumulative"] - summary["step_count"]
    return summary


def _lookup_history(
    summary: pd.DataFrame,
    entity: str,
    keys: pd.Series,
    steps: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    requests = pd.DataFrame(
        {entity: keys.astype(object).to_numpy(), "step": steps.to_numpy()}
    ).drop_duplicates()
    right = summary[[entity, "step", "cumulative"]].rename(
        columns={"step": "history_step"}
    ).sort_values("history_step")

    def cumulative_at_offset(offset: int) -> pd.Series:
        left = requests.copy()
        left["cutoff"] = left["step"] - offset
        left = left.sort_values("cutoff")
        matched = pd.merge_asof(
            left,
            right,
            left_on="cutoff",
            right_on="history_step",
            by=entity,
            direction="backward",
        )
        return matched.set_index([entity, "step"])["cumulative"].fillna(0)

    before = cumulative_at_offset(1)
    before_window = cumulative_at_offset(25)
    requests_index = pd.MultiIndex.from_frame(requests[[entity, "step"]])
    before = before.reindex(requests_index).to_numpy()
    before_window = before_window.reindex(requests_index).to_numpy()
    row_keys = pd.MultiIndex.from_arrays([keys.to_numpy(), steps.to_numpy()])
    before_by_row = pd.Series(before, index=requests_index).reindex(row_keys).to_numpy()
    window_by_row = pd.Series(
        (before - before_window).clip(min=0), index=requests_index
    ).reindex(row_keys).to_numpy()
    return before_by_row, window_by_row


def build_network_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return network role-degree features in original row order.

    The entity intersection only limits work to accounts that occur in both
    roles somewhere in the reference data. Feature values themselves are
    computed from earlier steps only; current-step and future edges are
    never used.
    """
    required = {"step", "nameOrig", "nameDest"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing network input columns: {sorted(missing)}")
    if not transactions["step"].is_monotonic_increasing:
        raise ValueError("Network features require rows in nondecreasing step order")

    origins = pd.Index(transactions["nameOrig"].dropna().unique())
    destinations = pd.Index(transactions["nameDest"].dropna().unique())
    shared_entities = origins.intersection(destinations)
    del origins, destinations

    out = pd.DataFrame(0, index=transactions.index, columns=NETWORK_FEATURES, dtype="int32")
    if len(shared_entities) == 0:
        return out

    step = transactions["step"]
    sender_mask = transactions["nameOrig"].isin(shared_entities)
    receiver_mask = transactions["nameDest"].isin(shared_entities)

    if sender_mask.any():
        incoming_events = transactions.loc[
            transactions["nameDest"].isin(shared_entities), ["nameDest", "step"]
        ]
        history = _entity_history(incoming_events, "nameDest")
        prior, last24 = _lookup_history(
            history,
            "nameDest",
            transactions.loc[sender_mask, "nameOrig"],
            step.loc[sender_mask],
        )
        out.loc[sender_mask, "network_sender_prior_in_degree"] = prior.astype("int32")
        out.loc[sender_mask, "network_sender_in_degree_last24"] = last24.astype("int32")
        del incoming_events, history

    if receiver_mask.any():
        outgoing_events = transactions.loc[
            transactions["nameOrig"].isin(shared_entities), ["nameOrig", "step"]
        ]
        history = _entity_history(outgoing_events, "nameOrig")
        prior, last24 = _lookup_history(
            history,
            "nameOrig",
            transactions.loc[receiver_mask, "nameDest"],
            step.loc[receiver_mask],
        )
        out.loc[receiver_mask, "network_receiver_prior_out_degree"] = prior.astype("int32")
        out.loc[receiver_mask, "network_receiver_out_degree_last24"] = last24.astype("int32")
        del outgoing_events, history

    return out


def build_investigation_network(
    reference_path: str | Path,
    *,
    step: int,
    sender: str,
    receiver: str,
    window_steps: int = 24,
    max_edges_per_direction: int = 8,
) -> dict[str, Any]:
    """Build a bounded NetworkX ego-context from strictly earlier edges."""
    if window_steps < 1 or max_edges_per_direction < 1:
        raise ValueError("window_steps and max_edges_per_direction must be positive")

    history = pd.read_parquet(
        reference_path,
        columns=["step", "nameOrig", "nameDest", "amount"],
        filters=[("step", ">=", max(1, step - window_steps)), ("step", "<", step)],
    )
    focus = {sender, receiver}
    related = history.loc[
        history["nameOrig"].isin(focus) | history["nameDest"].isin(focus)
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(focus)
    for row in related.itertuples(index=False):
        if graph.has_edge(row.nameOrig, row.nameDest):
            edge = graph[row.nameOrig][row.nameDest]
            edge["transactions"] += 1
            edge["amount"] += float(row.amount)
            edge["last_step"] = int(row.step)
        else:
            graph.add_edge(
                row.nameOrig,
                row.nameDest,
                transactions=1,
                amount=float(row.amount),
                last_step=int(row.step),
            )

    def counterparties(node: str, direction: str) -> list[dict[str, Any]]:
        if direction == "incoming":
            edges = [(u, node, data) for u, _, data in graph.in_edges(node, data=True)]
        else:
            edges = [(node, v, data) for _, v, data in graph.out_edges(node, data=True)]
        edges.sort(key=lambda edge: (edge[2]["transactions"], edge[2]["last_step"]), reverse=True)
        return [
            {
                "account": u if direction == "incoming" else v,
                "transactions": int(data["transactions"]),
                "amount": round(float(data["amount"]), 2),
                "last_seen_step": int(data["last_step"]),
            }
            for u, v, data in edges[:max_edges_per_direction]
        ]

    return {
        "window_steps": window_steps,
        "history_before_step": step,
        "observed_edge_count": graph.number_of_edges(),
        "prior_relationship_seen": bool(graph.has_edge(sender, receiver)),
        "sender": {
            "account": sender,
            "prior_in_degree": int(graph.in_degree[sender]),
            "prior_out_degree": int(graph.out_degree[sender]),
            "prior_receivers": counterparties(sender, "outgoing"),
            "prior_senders": counterparties(sender, "incoming"),
        },
        "receiver": {
            "account": receiver,
            "prior_in_degree": int(graph.in_degree[receiver]),
            "prior_out_degree": int(graph.out_degree[receiver]),
            "prior_receivers": counterparties(receiver, "outgoing"),
            "prior_senders": counterparties(receiver, "incoming"),
        },
    }
