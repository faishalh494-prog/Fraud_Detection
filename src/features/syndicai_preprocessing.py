"""
SyndicAI V1 — PaySim preprocessing & feature engineering pipeline
===================================================================

Reproducible, leakage-safe pipeline that turns the raw PaySim CSV into a
processed dataset for two model inputs:

  MODEL A = transaction-level features only
  MODEL B = MODEL A + leakage-safe behavioural features

Hard rules enforced by this script (do not relax without re-reading the
"WHY" comments — each one maps to a concrete, measured property of this
specific dataset, not a generic assumption):

  1. The raw CSV is never modified. It is only read.
  2. No rows are ever dropped or shuffled.
  3. All behavioural aggregates for a transaction use ONLY transactions
     with a strictly SMALLER `step` value. Transactions sharing the same
     `step` as the current one are never used to describe it — because
     `step` is an hour-level bucket (avg ~8,563 transactions/step, max
     51,352) and there is no reliable sub-hour ordering in the raw data,
     so two transactions in the same step cannot be safely ordered
     relative to one another.
  4. Train/val/test are split chronologically by `step`, never randomly.

Usage
-----
    python syndicai_preprocessing.py \
        --input /path/to/PS_20174392719_1491204439457_log.csv \
        --outdir ./syndicai_v1_output

Rerunning this script on the same raw CSV reproduces byte-identical
outputs (no randomness anywhere in the pipeline).
"""

import argparse
import ctypes
import gc
import json
from importlib import import_module
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import pandas as pd

resource = None
try:
    resource = import_module("resource")
except ModuleNotFoundError:
    pass


def _windows_peak_rss_mb() -> float:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel32.GetCurrentProcess()
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    success = psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    if not success:
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / (1024 * 1024)


def _peak_rss_mb() -> float:
    if resource is None:
        return _windows_peak_rss_mb()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _log_mem(label: str) -> None:
    print(f"      [mem] peak RSS so far: {_peak_rss_mb():.0f} MB  ({label})")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string[pyarrow]",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string[pyarrow]",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

RAW_COLUMNS = list(RAW_DTYPES.keys())

# Rolling-velocity window, in steps. step is an hour bucket, so 24 = 1 day.
VELOCITY_WINDOW_STEPS = 24

# Time-based split boundaries, chosen by inspecting cumulative row/fraud
# counts by step on the actual data (see report). Chosen to keep every
# split on whole `step` boundaries (never splitting a step's rows across
# sets) while giving val/test enough fraud examples to evaluate on.
SPLIT_TRAIN_MAX_STEP = 323   # steps 1-323   -> train
SPLIT_VAL_MAX_STEP = 378     # steps 324-378 -> val
# steps 379-743 -> test

TOLERANCE = 1e-2  # currency-rounding tolerance for balance sanity checks


# ---------------------------------------------------------------------------
# Step 1: Load + inspect raw data (read-only)
# ---------------------------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=RAW_DTYPES)
    # Defensive: raw file must already be in the columns/order we assume
    assert list(df.columns) == RAW_COLUMNS, (
        f"Raw CSV columns changed unexpectedly: {list(df.columns)}"
    )
    return df


def inspect_raw_data(df: pd.DataFrame, file_name: str) -> dict:
    report = {}
    report["file_name"] = file_name
    report["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    report["columns"] = list(df.columns)
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

    miss = df.isna().sum()
    report["missing_values"] = {c: int(v) for c, v in miss.items() if v > 0}
    report["missing_values_total"] = int(miss.sum())

    report["duplicate_rows_full"] = int(df.duplicated(keep=False).sum())

    vc = df["isFraud"].value_counts().sort_index()
    report["isFraud_distribution"] = {int(k): int(v) for k, v in vc.items()}
    report["isFraud_rate_pct"] = float(df["isFraud"].mean() * 100)

    report["isFlaggedFraud_distribution"] = {
        int(k): int(v) for k, v in df["isFlaggedFraud"].value_counts().sort_index().items()
    }

    report["transaction_types"] = {str(k): int(v) for k, v in df["type"].value_counts().items()}
    report["fraud_count_by_type"] = {
        str(k): int(v)
        for k, v in df.loc[df["isFraud"] == 1, "type"].value_counts().items()
    }

    report["step_min"] = int(df["step"].min())
    report["step_max"] = int(df["step"].max())
    report["step_is_sorted_nondecreasing"] = bool((df["step"].diff().dropna() >= 0).all())

    report["nameOrig_unique"] = int(df["nameOrig"].nunique())
    report["nameDest_unique"] = int(df["nameDest"].nunique())
    orig_counts = df["nameOrig"].value_counts()
    report["nameOrig_appearing_more_than_once"] = int((orig_counts > 1).sum())
    report["nameOrig_max_repeat"] = int(orig_counts.max())
    dest_counts = df["nameDest"].value_counts()
    report["nameDest_appearing_more_than_once"] = int((dest_counts > 1).sum())
    report["nameDest_max_repeat"] = int(dest_counts.max())

    report["negative_value_rows"] = {
        col: int((df[col] < 0).sum())
        for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    }
    report["zero_amount_rows"] = int((df["amount"] == 0).sum())

    dest_is_merchant = df["nameDest"].str.startswith("M")
    report["merchant_dest_rows_total"] = int(dest_is_merchant.sum())
    report["merchant_dest_with_nonzero_balance_rows"] = int(
        (dest_is_merchant & ((df["oldbalanceDest"] != 0) | (df["newbalanceDest"] != 0))).sum()
    )

    report["amount_describe"] = {k: float(v) for k, v in df["amount"].describe().items()}

    return report


# ---------------------------------------------------------------------------
# Step 2: Transaction-level features (Model A). Row-independent, zero
# cross-row aggregation -> zero leakage risk by construction.
# ---------------------------------------------------------------------------

def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # NOTE: mutates and returns the same object (adds columns only, never
    # touches existing raw columns) rather than df.copy() -- with a
    # 6.36M-row frame, an extra full copy is a real, avoidable memory cost
    # in a constrained environment, and every raw column value is only
    # ever assigned to, never read-modified-written, so in-place is safe.
    out = df

    # --- Temporal, derived purely from this row's own `step` ---
    out["hour_of_day"] = (out["step"] % 24).astype("int16")
    out["day_of_sim"] = (out["step"] // 24 + 1).astype("int16")
    out["is_night"] = out["hour_of_day"].between(0, 5).astype("int8")

    # --- Amount, this row only ---
    out["amount_log1p"] = np.log1p(out["amount"]).astype("float32")

    # --- One-hot type flags (kept alongside the categorical `type` column
    # so both tree-based and linear/sklearn models can consume it directly) ---
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        out[f"type_{t}"] = (out["type"] == t).astype("int8")

    # -----------------------------------------------------------------
    # DIAGNOSTIC / HIGH-LEAKAGE-RISK GROUP — computed for transparency,
    # kept in the dataset, but EXCLUDED from MODEL_A_FEATURES /
    # MODEL_B_FEATURES by default. See report section "PaySim balance
    # artefacts" for the numbers behind this decision:
    #   - amount == oldbalanceOrg happens in 0.00% of legit TRANSFER/
    #     CASH_OUT rows vs 97.82% of fraud rows in the same subset.
    #     This is not a discovered fraud *pattern*, it is PaySim's own
    #     fraud-injection rule leaking directly into a feature. A model
    #     trained on it isn't learning fraud behaviour, it's decoding
    #     the simulator.
    #   - oldbalanceDest == newbalanceDest == 0 happens in 0.06% of
    #     legit vs 49.63% of fraud TRANSFER/CASH_OUT rows (>800x lift) —
    #     the merchant-balance-not-tracked artefact bleeding into
    #     customer-to-customer transfers for fraud rows specifically.
    # -----------------------------------------------------------------
    out["diagnostic_amt_to_oldbalanceOrg_ratio"] = (
        out["amount"] / (out["oldbalanceOrg"] + 1.0)
    ).astype("float32")
    out["diagnostic_orig_balance_delta"] = (
        out["oldbalanceOrg"] - out["amount"] - out["newbalanceOrig"]
    ).astype("float32")
    dest_is_merchant = out["nameDest"].str.startswith("M")
    dest_delta = out["newbalanceDest"] - out["oldbalanceDest"] - out["amount"]
    out["diagnostic_dest_balance_delta"] = dest_delta.where(~dest_is_merchant, np.nan).astype("float32")
    out["diagnostic_orig_fully_drained"] = (
        (out["newbalanceOrig"] == 0) & (out["oldbalanceOrg"] > 0)
    ).astype("int8")
    out["diagnostic_dest_zero_zero_balance"] = (
        (~dest_is_merchant) & (out["oldbalanceDest"] == 0) & (out["newbalanceDest"] == 0)
    ).astype("int8")

    return out


# ---------------------------------------------------------------------------
# Step 3: Leakage-safe behavioural features (Model B additions)
# ---------------------------------------------------------------------------

def _build_entity_step_summary(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Per (entity, step): count/sum THIS step, plus cumulative-BEFORE-this-step
    aggregates. All rows sharing an (entity, step) pair get identical
    'before' values by construction — this is deliberate: we cannot order
    transactions within the same step, so we never let them see each other.
    """
    sl = (
        df.groupby([entity_col, "step"], observed=True)
        .agg(step_count=("amount", "size"), step_amount_sum=("amount", "sum"))
        .reset_index()
        .sort_values([entity_col, "step"])
        .reset_index(drop=True)
    )
    sl["step_amount_sum"] = sl["step_amount_sum"].astype("float32")
    g = sl.groupby(entity_col, observed=True)
    sl["cum_count_incl"] = g["step_count"].cumsum()
    sl["cum_amount_incl"] = g["step_amount_sum"].cumsum().astype("float32")
    sl["count_before"] = sl["cum_count_incl"] - sl["step_count"]
    sl["amount_sum_before"] = (sl["cum_amount_incl"] - sl["step_amount_sum"]).astype("float32")
    sl["prev_step"] = g["step"].shift(1)
    return sl


def _add_rolling_velocity(sl: pd.DataFrame, entity_col: str, window: int) -> pd.Series:
    """For each (entity, step) row in the compact step-level summary `sl`,
    compute count of prior transactions strictly within the trailing
    `window` steps (i.e. step in [current_step - window, current_step - 1]).
    Implemented as a difference of two cumulative-before lookups, using
    merge_asof for the second lookup since (current_step - window) is
    generally not one of the entity's own observed step values.
    Validated against hand-computed examples before use on full data.
    """
    left = sl[[entity_col, "step"]].copy()
    left["asof_key"] = left["step"] - window - 1
    left = left.sort_values("asof_key")
    right = sl[[entity_col, "step", "cum_count_incl"]].sort_values("step")

    asof = pd.merge_asof(
        left, right, left_on="asof_key", right_on="step", by=entity_col,
        direction="backward", suffixes=("", "_r"),
    )
    asof["count_before_window_start"] = asof["cum_count_incl"].fillna(0)
    asof = asof.sort_index()  # restore original sl row order (index preserved by merge_asof's left)

    # sl and asof share the same row identity via entity_col+step; join back explicitly to be safe
    merged = sl[[entity_col, "step", "count_before"]].merge(
        asof[[entity_col, "step", "count_before_window_start"]], on=[entity_col, "step"], how="left"
    )
    velocity = (merged["count_before"] - merged["count_before_window_start"]).clip(lower=0)
    return velocity.astype("int32").values


def build_receiver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features describing nameDest's history BEFORE this
    transaction. Receiver-centric because the data shows receivers, unlike
    senders, genuinely repeat: within the fraud-relevant TRANSFER/CASH_OUT
    subset, 381,234 / 509,565 (75%) of destination accounts appear more
    than once (max 75x) — consistent with mule/cash-out-agent typology,
    a real structural signal, not a simulator quirk.

    NOTE on a feature we deliberately do NOT build: "unique previous
    senders to this receiver". Checked directly against the data first:
    every (nameDest, nameOrig) pair in the entire 6,362,620-row file is
    unique -- zero sender ever sends to the same receiver twice, anywhere,
    at any time. That means unique_senders_before would be mathematically
    identical to receiver_txn_count_before for every single row, adding a
    column with zero new information at a very high computational cost
    (the groupby underlying it does not compress at all, since almost
    every (nameDest, nameOrig) pair is already unique). Dropped as
    redundant, per "keep only features that make sense".
    """
    sl = _build_entity_step_summary(df, "nameDest")
    sl["velocity_last24_before"] = _add_rolling_velocity(sl, "nameDest", VELOCITY_WINDOW_STEPS)

    # Look up each row's (nameDest, step) in the compact summary WITHOUT a
    # full df.merge(): merge() always builds an entirely new frame with a
    # copy of every existing column, which briefly doubles memory for the
    # whole (large) dataframe just to attach a handful of new columns.
    # reindex() against a MultiIndex fetches only the requested columns,
    # in df's row order, at a fraction of the memory cost.
    value_cols = ["count_before", "amount_sum_before", "prev_step", "velocity_last24_before"]
    lookup = sl.set_index(["nameDest", "step"])[value_cols]
    key = pd.MultiIndex.from_arrays([df["nameDest"].values, df["step"].values])
    fetched = lookup.reindex(key)
    del sl, lookup, key
    gc.collect()

    out = df
    out["receiver_txn_count_before"] = fetched["count_before"].astype("int32").values
    out["receiver_total_amount_before"] = fetched["amount_sum_before"].astype("float32").values
    out["receiver_txn_count_last24_before"] = fetched["velocity_last24_before"].astype("int32").values
    prev_step_vals = fetched["prev_step"].values
    del fetched
    gc.collect()

    out["receiver_avg_amount_before"] = (
        out["receiver_total_amount_before"] / out["receiver_txn_count_before"].replace(0, np.nan)
    ).fillna(0.0).astype("float32")
    out["receiver_steps_since_last"] = pd.Series(out["step"].values - prev_step_vals, index=out.index).fillna(-1).astype("int32")
    out["receiver_is_new"] = (out["receiver_txn_count_before"] == 0).astype("int8")

    gc.collect()
    return out


def build_sender_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sender (nameOrig) history BEFORE this transaction.
    Kept minimal and deliberately: 6,353,307 / 6,362,620 nameOrig values
    (99.85%) appear EXACTLY ONCE in the entire raw file, and of the 8,213
    fraud transactions, only 28 originate from a sender who ever
    transacts more than once anywhere in the dataset. Sender-side
    lifetime average/median/unique-beneficiary features would therefore
    be computed from 0-2 prior points for virtually every row -- mostly
    undefined/near-constant and not worth the added columns. We keep only
    count + is_new, which are cheap, well-defined for every row, and let
    the model see "this specific customer ID has (or hasn't) sent money
    before" without inventing statistics over near-empty history.

    Implementation deliberately avoids a full (nameOrig, step) groupby:
    since nameOrig is a near-fully-unique key, that groupby barely
    compresses at all (it's essentially a full-size copy of the data --
    the same cost that caused an OOM when the equivalent computation was
    still present on the receiver side). Instead: find the small set of
    nameOrig values that repeat ANYWHERE in the file (0.15% of them), do
    the leak-safe cumulative computation only on that small subset, and
    default every other row to 0 / is_new=1 directly -- which is exactly
    correct, since a sender who appears exactly once in the whole dataset
    has no prior transaction by construction.
    """
    orig_counts = df["nameOrig"].value_counts()
    repeat_origs = orig_counts[orig_counts > 1].index
    out = df
    out["sender_txn_count_before"] = np.int32(0)

    if len(repeat_origs) > 0:
        repeat_mask = out["nameOrig"].isin(repeat_origs)
        small = out.loc[repeat_mask, ["nameOrig", "step", "amount"]]
        sl = _build_entity_step_summary(small, "nameOrig")
        lookup = small[["nameOrig", "step"]].merge(
            sl[["nameOrig", "step", "count_before"]], on=["nameOrig", "step"], how="left"
        )
        out.loc[repeat_mask, "sender_txn_count_before"] = lookup["count_before"].astype("int32").values
        del small, sl, lookup, repeat_mask
        gc.collect()

    out["sender_is_new"] = (out["sender_txn_count_before"] == 0).astype("int8")
    del orig_counts, repeat_origs
    gc.collect()
    return out


def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    out = build_receiver_features(df)
    out = build_sender_features(out)
    return out


# ---------------------------------------------------------------------------
# Step 4: Time-based split
# ---------------------------------------------------------------------------

def time_based_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df  # in-place: only adds a new column, see note in build_transaction_features
    conditions = [
        out["step"] <= SPLIT_TRAIN_MAX_STEP,
        (out["step"] > SPLIT_TRAIN_MAX_STEP) & (out["step"] <= SPLIT_VAL_MAX_STEP),
        out["step"] > SPLIT_VAL_MAX_STEP,
    ]
    out["split"] = np.select(conditions, ["train", "val", "test"], default="unassigned")
    assert (out["split"] != "unassigned").all(), "Every row must fall into train/val/test"
    return out


# ---------------------------------------------------------------------------
# Step 5: Validation checks (must all pass before outputs are trusted)
# ---------------------------------------------------------------------------

def run_validation_checks(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    checks = {}

    # 1. Row count preserved exactly, no dropping
    checks["row_count_preserved"] = bool(len(raw_df) == len(processed_df))

    # 2. isFraud target preserved identically (same values, same order)
    checks["isFraud_preserved_identical"] = bool(
        (raw_df["isFraud"].values == processed_df["isFraud"].values).all()
    )

    # 3. All raw columns still present, untouched, and in the exact same
    #    row order as the raw file (merges use a unique right-hand join
    #    key, which pandas guarantees preserves left-frame order, but we
    #    verify it directly rather than assume it). pd.Series.equals() is
    #    used instead of manual array casting: it's implemented in
    #    C/Cython and does not materialize a second copy of the string
    #    columns as Python object arrays, which is what caused an OOM
    #    kill the first time this check was written naively.
    raw_reset = raw_df.reset_index(drop=True)
    proc_reset = processed_df.reset_index(drop=True)
    checks["row_order_preserved"] = bool(raw_reset["step"].equals(proc_reset["step"]))
    raw_untouched = all(raw_reset[c].equals(proc_reset[c]) for c in RAW_COLUMNS)
    checks["raw_columns_untouched"] = bool(raw_untouched)

    # 4. Chronological split: max(train.step) <= min(val.step) and max(val.step) <= min(test.step)
    tr = processed_df.loc[processed_df["split"] == "train", "step"]
    va = processed_df.loc[processed_df["split"] == "val", "step"]
    te = processed_df.loc[processed_df["split"] == "test", "step"]
    checks["split_is_chronological"] = bool(tr.max() < va.min() <= va.max() < te.min())
    checks["split_row_counts"] = {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))}
    checks["split_fraud_counts"] = {
        "train": int(processed_df.loc[processed_df["split"] == "train", "isFraud"].sum()),
        "val": int(processed_df.loc[processed_df["split"] == "val", "isFraud"].sum()),
        "test": int(processed_df.loc[processed_df["split"] == "test", "isFraud"].sum()),
    }
    checks["split_fraud_rate_pct"] = {
        k: round(v / checks["split_row_counts"][k] * 100, 4)
        for k, v in checks["split_fraud_counts"].items()
    }

    # 5. Behavioural "before" counts can never exceed how many prior rows
    #    of that entity actually exist strictly before this step, and must
    #    be non-negative.
    checks["receiver_count_before_nonnegative"] = bool((processed_df["receiver_txn_count_before"] >= 0).all())
    checks["sender_count_before_nonnegative"] = bool((processed_df["sender_txn_count_before"] >= 0).all())
    checks["velocity_leq_lifetime_count"] = bool(
        (processed_df["receiver_txn_count_last24_before"] <= processed_df["receiver_txn_count_before"]).all()
    )

    # 6. Direct leakage probe: for every row where a receiver's FIRST-EVER
    #    appearance is exactly at this row's step, receiver_txn_count_before
    #    must be 0. (Cannot know about itself or same-step siblings.)
    first_step_per_dest = processed_df.groupby("nameDest")["step"].transform("min")
    is_first_step_row = processed_df["step"] == first_step_per_dest
    checks["new_receiver_has_zero_prior_count"] = bool(
        (processed_df.loc[is_first_step_row, "receiver_txn_count_before"] == 0).all()
    )

    # 7. Gold-standard leakage test: recompute behavioural features for a
    #    random sample of rows using ONLY the raw data truncated to
    #    step < that row's step, and confirm exact match with the pipeline
    #    output. This directly proves "before" features do not use any
    #    same-step or future information, by reconstruction rather than
    #    by trusting the implementation.
    # Pick a handful of random step values and, for EACH, independently
    # recompute "count of prior transactions per entity" from the raw file
    # truncated to step < that value, then check EVERY processed row at
    # that step against the recomputation. This validates far more rows
    # than a per-row random sample while only paying the full-frame
    # filter cost ~20 times instead of ~200.
    rng = np.random.default_rng(42)
    raw_small = raw_df[["step", "nameOrig", "nameDest"]]
    candidate_steps = processed_df["step"].unique()
    chosen_steps = rng.choice(candidate_steps, size=min(12, len(candidate_steps)), replace=False)

    mismatches = 0
    rows_checked = 0
    for step_val in chosen_steps:
        past = raw_small[raw_small["step"] < step_val]
        past_dest_counts = past["nameDest"].value_counts()
        past_orig_counts = past["nameOrig"].value_counts()
        current = processed_df.loc[
            processed_df["step"] == step_val,
            ["nameDest", "nameOrig", "receiver_txn_count_before", "sender_txn_count_before"],
        ]
        expected_recv = current["nameDest"].map(past_dest_counts).fillna(0).astype(int)
        expected_send = current["nameOrig"].map(past_orig_counts).fillna(0).astype(int)
        mismatches += int((expected_recv.values != current["receiver_txn_count_before"].values).sum())
        mismatches += int((expected_send.values != current["sender_txn_count_before"].values).sum())
        rows_checked += len(current)
        del past, past_dest_counts, past_orig_counts, current
    checks["gold_standard_recompute_steps_sampled"] = int(len(chosen_steps))
    checks["gold_standard_recompute_rows_checked"] = int(rows_checked)
    checks["gold_standard_recompute_mismatches"] = int(mismatches)
    checks["gold_standard_recompute_passed"] = bool(mismatches == 0)

    checks["all_passed"] = bool(all(
        v for k, v in checks.items()
        if isinstance(v, bool)
    ))
    return checks


# ---------------------------------------------------------------------------
# Feature group definitions (what each model should actually be trained on)
# ---------------------------------------------------------------------------

MODEL_A_FEATURES = [
    "step", "hour_of_day", "day_of_sim", "is_night",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    "amount", "amount_log1p",
]

MODEL_B_FEATURES = MODEL_A_FEATURES + [
    "receiver_txn_count_before", "receiver_total_amount_before", "receiver_avg_amount_before",
    "receiver_steps_since_last", "receiver_is_new",
    "receiver_txn_count_last24_before",
    "sender_txn_count_before", "sender_is_new",
]

# Available but NOT in the default lists above -- see docstring in
# build_transaction_features(). Use only for deliberate ablation studies.
DIAGNOSTIC_FEATURES = [
    "diagnostic_amt_to_oldbalanceOrg_ratio",
    "diagnostic_orig_balance_delta",
    "diagnostic_dest_balance_delta",
    "diagnostic_orig_fully_drained",
    "diagnostic_dest_zero_zero_balance",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]


def load_model_inputs(df: pd.DataFrame, model: str = "A", split: str | None = None) -> pd.DataFrame:
    """Convenience loader: returns (X, y) columns for MODEL A or MODEL B,
    optionally filtered to one split ('train'/'val'/'test')."""
    cols = MODEL_A_FEATURES if model.upper() == "A" else MODEL_B_FEATURES
    d = df if split is None else df[df["split"] == split]
    return d[cols].copy(), d["isFraud"].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SyndicAI V1 preprocessing pipeline")
    parser.add_argument("--input", required=True, help="Path to raw PaySim CSV")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/7] Loading raw data from {args.input} ...")
    raw_df = load_raw_data(args.input)
    print(f"      Loaded {len(raw_df):,} rows in {time.time()-t0:.1f}s")
    _log_mem("after load")

    print("[2/7] Inspecting raw data ...")
    raw_report = inspect_raw_data(raw_df, Path(args.input).name)
    with open(outdir / "raw_data_inspection.json", "w") as f:
        json.dump(raw_report, f, indent=2, default=str)
    _log_mem("after inspect")

    print("[3/7] Building transaction-level features (Model A) ...")
    t1 = time.time()
    df = build_transaction_features(raw_df)
    del raw_df  # freed: we mutated in place, this name is now stale; a
    # fresh copy is reloaded from disk later, specifically for validation,
    # rather than keeping a second full-size frame alive for the whole
    # (expensive) feature-engineering phase.
    print(f"      Done in {time.time()-t1:.1f}s")
    _log_mem("after transaction features")

    print("[4/7] Building leakage-safe behavioural features (Model B additions) ...")
    t1 = time.time()
    df = build_behavioral_features(df)
    gc.collect()
    print(f"      Done in {time.time()-t1:.1f}s")
    _log_mem("after behavioural features")

    print("[5/7] Applying time-based split ...")
    df = time_based_split(df)
    _log_mem("after split")

    print("[6/6] Saving outputs ...")
    t1 = time.time()
    df.to_parquet(outdir / "syndicai_v1_processed.parquet", index=False)
    for split_name in ["train", "val", "test"]:
        df[df["split"] == split_name].to_parquet(
            outdir / f"syndicai_v1_{split_name}.parquet", index=False
        )
    df.head(2000).to_csv(outdir / "syndicai_v1_processed_preview.csv", index=False)
    print(f"      Done in {time.time()-t1:.1f}s")
    _log_mem("after saving outputs")

    print(f"\nFeature engineering done in {time.time()-t0:.1f}s total. Outputs written to {outdir}/")
    print("Run validate_output.py as a SEPARATE process to check for leakage --")
    print("keeping it separate avoids holding the raw file, the fully-featured")
    print("dataframe, and a second raw reload all alive in one process at once.")


if __name__ == "__main__":
    main()
