"""
SyndicAI V1 -- standalone, memory-lean validation pass
========================================================

Run AFTER syndicai_preprocessing.py has produced
syndicai_v1_processed.parquet. Every check below loads ONLY the specific
columns it needs (via read_csv usecols= / read_parquet columns=) and
discards them before the next check, rather than holding the full raw
file and the full processed file (34 columns each) in memory
simultaneously. This is deliberate: on a memory-constrained machine,
loading everything at once is what caused repeated OOM kills during
development; loading narrow slices per-check does not, and pandas can
re-read a 6.36M-row CSV in ~10s, which is a fine trade for reliability.

Usage:
    python validate_output.py --input <raw_csv> --processed <processed_parquet> --outdir <dir>
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from syndicai_preprocessing import RAW_DTYPES, RAW_COLUMNS


def load_raw_cols(path, cols):
    dtypes = {c: RAW_DTYPES[c] for c in cols}
    return pd.read_csv(path, usecols=cols, dtype=dtypes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    checks = {}

    print("[1/7] row count / row order ...")
    raw_step = load_raw_cols(args.input, ["step"])
    proc_step_only = pd.read_parquet(args.processed, columns=["step"])
    checks["row_count_preserved"] = bool(len(raw_step) == len(proc_step_only))
    checks["row_order_preserved"] = bool(
        raw_step["step"].reset_index(drop=True).equals(proc_step_only["step"].reset_index(drop=True))
    )
    del raw_step, proc_step_only
    gc.collect()

    print("[2/7] isFraud preserved ...")
    raw_fraud = load_raw_cols(args.input, ["isFraud"])
    proc_fraud = pd.read_parquet(args.processed, columns=["isFraud"])
    checks["isFraud_preserved_identical"] = bool(
        (raw_fraud["isFraud"].reset_index(drop=True).values == proc_fraud["isFraud"].reset_index(drop=True).values).all()
    )
    del raw_fraud, proc_fraud
    gc.collect()

    print("[3/7] raw columns untouched, one column at a time ...")
    col_results = {}
    for c in RAW_COLUMNS:
        raw_c = load_raw_cols(args.input, [c])[c].reset_index(drop=True)
        proc_c = pd.read_parquet(args.processed, columns=[c])[c].reset_index(drop=True)
        col_results[c] = bool(raw_c.equals(proc_c))
        del raw_c, proc_c
        gc.collect()
    checks["raw_columns_untouched_per_column"] = col_results
    checks["raw_columns_untouched"] = bool(all(col_results.values()))

    print("[4/7] split checks ...")
    proc_split = pd.read_parquet(args.processed, columns=["step", "split", "isFraud"])
    tr = proc_split.loc[proc_split["split"] == "train", "step"]
    va = proc_split.loc[proc_split["split"] == "val", "step"]
    te = proc_split.loc[proc_split["split"] == "test", "step"]
    checks["split_is_chronological"] = bool(tr.max() < va.min() <= va.max() < te.min())
    checks["split_row_counts"] = {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))}
    checks["split_fraud_counts"] = {
        s: int(proc_split.loc[proc_split["split"] == s, "isFraud"].sum()) for s in ["train", "val", "test"]
    }
    checks["split_fraud_rate_pct"] = {
        k: round(v / checks["split_row_counts"][k] * 100, 4) for k, v in checks["split_fraud_counts"].items()
    }
    del proc_split, tr, va, te
    gc.collect()

    print("[5/7] non-negativity / bounds on behavioural columns ...")
    behav_cols = ["receiver_txn_count_before", "sender_txn_count_before", "receiver_txn_count_last24_before"]
    b = pd.read_parquet(args.processed, columns=behav_cols)
    checks["receiver_count_before_nonnegative"] = bool((b["receiver_txn_count_before"] >= 0).all())
    checks["sender_count_before_nonnegative"] = bool((b["sender_txn_count_before"] >= 0).all())
    checks["velocity_leq_lifetime_count"] = bool(
        (b["receiver_txn_count_last24_before"] <= b["receiver_txn_count_before"]).all()
    )
    del b
    gc.collect()

    print("[6/7] new-receiver-has-zero-prior-count ...")
    p = pd.read_parquet(args.processed, columns=["nameDest", "step", "receiver_txn_count_before"])
    first_step_per_dest = p.groupby("nameDest")["step"].transform("min")
    is_first = p["step"] == first_step_per_dest
    checks["new_receiver_has_zero_prior_count"] = bool((p.loc[is_first, "receiver_txn_count_before"] == 0).all())
    del p, first_step_per_dest, is_first
    gc.collect()

    print("[7/7] gold-standard recompute ...")
    raw_small = load_raw_cols(args.input, ["step", "nameOrig", "nameDest"])
    proc_small = pd.read_parquet(
        args.processed, columns=["step", "nameOrig", "nameDest", "receiver_txn_count_before", "sender_txn_count_before"]
    )
    rng = np.random.default_rng(42)
    candidate_steps = proc_small["step"].unique()
    chosen_steps = rng.choice(candidate_steps, size=min(12, len(candidate_steps)), replace=False)

    mismatches = 0
    rows_checked = 0
    for step_val in chosen_steps:
        past = raw_small[raw_small["step"] < step_val]
        past_dest_counts = past["nameDest"].value_counts()
        past_orig_counts = past["nameOrig"].value_counts()
        current = proc_small[proc_small["step"] == step_val]
        expected_recv = current["nameDest"].map(past_dest_counts).fillna(0).astype(int)
        expected_send = current["nameOrig"].map(past_orig_counts).fillna(0).astype(int)
        mismatches += int((expected_recv.values != current["receiver_txn_count_before"].values).sum())
        mismatches += int((expected_send.values != current["sender_txn_count_before"].values).sum())
        rows_checked += len(current)
        del past, past_dest_counts, past_orig_counts, current
    del raw_small, proc_small
    gc.collect()

    checks["gold_standard_recompute_steps_sampled"] = int(len(chosen_steps))
    checks["gold_standard_recompute_rows_checked"] = int(rows_checked)
    checks["gold_standard_recompute_mismatches"] = int(mismatches)
    checks["gold_standard_recompute_passed"] = bool(mismatches == 0)

    checks["all_passed"] = bool(all(v for k, v in checks.items() if isinstance(v, bool)))

    with open(outdir / "validation_checks.json", "w") as f:
        json.dump(checks, f, indent=2, default=str)
    print(json.dumps(checks, indent=2, default=str))

    if not checks["all_passed"]:
        print("!! VALIDATION FAILED -- see validation_checks.json !!", file=sys.stderr)
        sys.exit(1)
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
