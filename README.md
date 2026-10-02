# SyndicAI V4

SyndicAI is an explainable fraud-investigation product for transaction-monitoring
analysts. A model score prioritizes human review; it is not a finding that a
transaction or person is fraudulent.

## Validated data foundation

V4 uses the existing leakage-checked V1 preprocessing outputs without rebuilding
them. Place the four existing files in `data/processed/`:

- `syndicai_v1_train.parquet`
- `syndicai_v1_val.parquet`
- `syndicai_v1_test.parquet`
- `syndicai_v1_processed.parquet`

The processed reference and split files are local data artifacts and are not
committed to Git.

## Train and compare

From the repository root with dependencies from `requirements.txt` installed:

```powershell
python -m src.syndicai_v4.train
```

The trainer uses the fixed chronological splits and fits three class-imbalance
weighted XGBoost models: A (transaction), B (transaction + prior behaviour), and
C (transaction + prior behaviour + prior network degree). The review threshold
for each model is selected by maximum F1 on validation data. Precision, recall,
F1, PR-AUC, false positives, and alert burden on the test period are written to
`models/metrics.json`. Fitted models, the derived network feature table, and
test-period alert queues are generated under the ignored `models/` directory.
The measured comparison from the supplied local dataset is summarized in
[docs/SYNDICAI_V4_EVALUATION.md](docs/SYNDICAI_V4_EVALUATION.md). In the recorded
run, B and C tied on held-out metrics and none of C's network features was used
in a tree split, so the investigator desk defaults to B rather than assuming
the graph inputs help.

Do not use test metrics to choose a threshold or tune the model. The included
test-period queue is a demonstration/investigation feed after the final holdout
evaluation, not live production data.

### Offline research: normal-behaviour anomaly

The rejected anomaly experiment is maintained outside the production
`src/syndicai_v4` package. After training the V4 baseline, run:

```powershell
python -m experiments.normal_behaviour_anomaly.anomaly_experiment
```

Methodology and results are documented in
[experiments/normal_behaviour_anomaly/README.md](experiments/normal_behaviour_anomaly/README.md).
This offline experiment is not loaded by production inference or the
investigator interface.

## Run the product

Start the API and investigator desk in separate terminals:

```powershell
python -m uvicorn backend.app:app --reload
```

```powershell
python -m streamlit run frontend/streamlit_app.py
```

The Streamlit desk reads scores and investigation details from the FastAPI
service. Set `SYNDICAI_API_URL` to use a different API URL. Open
`http://127.0.0.1:8000/docs` for the API reference.

The `POST /score` endpoint continues to score a zero-based row from the
validated processed reference dataset.

### Transaction-time scoring prototype

`POST /score_transaction` scores a new PaySim-style event against the processed
reference history. For Model B, sender and receiver history queries are
restricted to rows with `step < request.step`; current-step and later events
cannot contribute. The request needs only the fields consumed by the existing
Model A/B feature definitions:

```json
{
  "step": 600,
  "type": "TRANSFER",
  "amount": 12500.0,
  "nameOrig": "C123456789",
  "nameDest": "C987654321",
  "model": "B"
}
```

The response includes the model score and review threshold, TreeSHAP reasons,
and computed sender/receiver behavioural evidence. `model` defaults to `B`;
`A` is also supported. Model C is not supported for new events because its
current network feature construction depends on full-reference account-role
membership, including information from later rows.

The endpoint reads history from the fixed processed V1 reference dataset. It
does not persist scored events into that history or update features as new
events arrive. Treat it as a transaction-time inference prototype: no latency,
throughput, or production-readiness claim is made.

Example response excerpt from the documented request (the full response also
includes remaining feature contributions and the model input feature dictionary):

```json
{
  "model": "B",
  "risk": {
    "score": 98.59,
    "score_kind": "model score; not a calibrated probability",
    "risk_level": "High review priority",
    "review_threshold": 97.69,
    "flagged_for_review": true
  },
  "explanation": {
    "method": "XGBoost TreeSHAP contributions",
    "summary": "Score-increasing model evidence: Transfer transaction=1.0; ...",
    "caveat": "Model evidence supports review; it does not establish fraud.",
    "reasons": [
      {
        "feature": "type_TRANSFER",
        "label": "Transfer transaction",
        "value": 1.0,
        "contribution": 2.7065,
        "direction": "increases"
      }
    ]
  },
  "behavioural_evidence": {
    "receiver_txn_count_before": 0,
    "receiver_total_amount_before": 0.0,
    "receiver_avg_amount_before": 0.0,
    "receiver_steps_since_last": -1,
    "receiver_is_new": 1,
    "receiver_txn_count_last24_before": 0,
    "sender_txn_count_before": 0,
    "sender_is_new": 1
  },
  "history_rule": "Only reference transactions with step strictly less than this event were used."
}
```

Feature parity was checked against one sampled row from each of the seven
Parquet row groups (7 transactions, 56 behavioural values). Counts, flags,
last-activity steps, and velocity counts matched exactly. There were 6 exact
float mismatches: 3 `receiver_total_amount_before` and 3 corresponding
`receiver_avg_amount_before` values. The largest absolute difference was
0.03125 currency units (0.015625 for an average); these are float32 cumulative
rounding differences between the stored V1 artifacts and recomputation from
their historical amount rows. No integer behavioural feature mismatched.
This limitation is reported rather than hidden; see the focused parity test
in `tests/test_online_scoring.py`.

## Tests

```powershell
python -m unittest discover -s tests -p "test_syndicai_v4.py"
python -m unittest discover -s tests -p "test_online_scoring.py"
python -m unittest discover -s tests
```

Tests cover time-causal graph features, model threshold/evaluation helpers,
TreeSHAP explanations, persisted investigation states, online feature parity,
and transaction-request validation. The existing
`tests/validate_output.py` is the separate validation tool for the original V1
preprocessing outputs.