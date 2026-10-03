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

From the repository root, create an environment and install dependencies:

```powershell
python -m venv .venv
python -m pip install -r requirements.txt -c constraints.txt
```

For reproducible builds matching the validated test environment, `constraints.txt` pins exact dependency versions.

Place the supplied validated Parquet files listed above in `data/processed/`.
Then train and evaluate the V4 models:

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
Model B also improved measured test precision, F1, and PR-AUC over Model A
while producing fewer alerts. NetworkX remains useful for showing prior
counterparties and relationship context in historical investigations, but the
recorded Model C comparison does not show predictive gain. The validation-only
alert-budget trade-offs are in the evaluation report; test alert burden differs
from its validation target, so it is not a guaranteed investigator workload.

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
On the recorded data, adding its anomaly score did not improve the full-training
Model B: B + anomaly test PR-AUC was 0.4850 versus 0.5093 for Model B, and the
anomaly-only signal had PR-AUC 0.0049. It therefore remains a separate
investigation research result, not a production risk feature.

## Run the product

The investigator desk reads scores and investigation details from the FastAPI
service. Set `SYNDICAI_API_URL` to use a different API URL. API requests require
the same `SYNDICAI_API_KEY` in both process environments; use a randomly
generated secret of at least 32 characters and keep it out of source code,
command arguments, screenshots, and logs. The limited `/health` readiness
endpoint and API schema pages are unauthenticated; investigation and scoring
operations require the key. Configure the secret through the runtime
environment, for example by generating it without printing it:

```powershell
$env:SYNDICAI_API_KEY = (python -c "import secrets; print(secrets.token_urlsafe(32))")
```

Configure the key in your local environment or secret manager.

Start the FastAPI application, which directly serves the primary institutional
analyst frontend at `http://127.0.0.1:8000/`:

```powershell
python -m uvicorn backend.app:app --reload
```

Optionally, the Streamlit monitoring desk remains available in a separate terminal:

```powershell
python -m streamlit run frontend/streamlit_app.py
```

The API
refuses protected requests when a sufficiently long key is not configured.
Structured request/action logs exclude the key, account identifiers, and event
IDs. Open `http://127.0.0.1:8000/docs` for the API reference.

## Live event monitor and stream demo

`POST /score_transaction` remains the single event-scoring path. The live
simulator submits each deterministic PaySim-style event over HTTP; the API
builds prior-only features, scores with Model B, explains the result, and then
persists both the event and its scoring evidence in local SQLite. The
**LIVE MONITOR** workflow polls the protected `/live/status` and `/live/events`
endpoints every two seconds while open. Events sharing a step remain excluded
from each other's features, and failed events are not added to state.

Start the producer in a separate terminal with the same API key:

```powershell
$env:SYNDICAI_API_KEY = "<same locally-held value>"
$env:SYNDICAI_API_URL = "http://127.0.0.1:8000"
python -m demo.live_stream --interval 1.0
```

The producer runs continuously until Ctrl+C. Use
`python -m demo.live_stream --interval 0.2 --max-events 4` for one four-event
demo. Its first four inputs demonstrate unseen history, developing history,
changed transaction behaviour, and a high-scoring CASH_OUT. The model scores,
review priority, and explanation are generated during each request; they are
not canned. Event IDs are unique and duplicates are rejected. The source
resumes after the maximum persisted step and immutable reference step.

Benchmark the same real HTTP scorer without an intentional delay:

```powershell
python -m demo.benchmark_live --events 50
```

This writes the sequential events to the local SQLite store and prints
feature, inference, explanation, state-write, end-to-end latency distributions,
and events/second, together with environment and sample metadata. Results are
local measurements, not future latency or workload guarantees. Full method,
measured results, container instructions, and IBM Z/LinuxONE limitations are
maintained in
[docs/LIVE_STREAM_AND_DEPLOYMENT.md](docs/LIVE_STREAM_AND_DEPLOYMENT.md).

For a local container deployment (Docker Compose required), set
`SYNDICAI_API_KEY` in the shell and run `docker compose up --build`. The
container keeps the processed reference data read-only and the local model /
SQLite directory writable. This generic container has not been built for or
validated on IBM Z/LinuxONE; see the deployment note for the current s390x
dependency blocker and external validation required.

The investigator desk separates historical row-based investigation from
**New transaction review**. The latter obtains the immutable reference maximum
step from `GET /transaction_limits`, submits valid new events through
`POST /score_transaction`, and displays the score, threshold, explanation,
behavioural evidence, and whether history was updated. Model B is selected by
default; Model A is also available. A successful event is immediately stored
in local online history and can contribute to a later event at a strictly
greater step. This dashboard does not implement scoring or feature logic.
The result distinguishes risk score, review priority, and evidence strength;
calibrated probability is explicitly unavailable. Model B coverage is labelled
New, Limited history, or Established history from prior sender/receiver event
counts. Established means at least five prior transactions for both accounts.
This transparent coverage tier is not a risk-confidence or fraud measure;
missing or limited history does not indicate low or high fraud risk. Model A
does not use behavioural history.

New transaction review keeps the existing validation maximum-F1 operating
point as its default. Model B also offers the five existing alert-budget
operating points, each selected from validation scores only. The API evaluates
and caches the selected thresholds against the held-out test period to show
historical false-positive and workload trade-offs. Selecting another point is
explicit; it changes only the new-event review threshold and does not change
model inputs, Model A/B/C evaluation, or historical `/score` results. Test
volume is historical evidence, not a future workload guarantee. Alternative
points are never auto-selected.

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
  "step": 800,
  "type": "TRANSFER",
  "amount": 12500.0,
  "nameOrig": "C123456789",
  "nameDest": "C987654321",
  "model": "B"
}
```

The response includes the risk score, review priority, selected review
threshold, TreeSHAP reasons, behavioural evidence, and a separate behavioural
history coverage status. The score is not a calibrated probability; the API
returns `calibrated_probability: null`. `model` defaults to `B`; `A` is also
supported. Model C is not supported for new events because its
current network feature construction depends on full-reference account-role
membership, including information from later rows.

The optional `operating_point` request field defaults to `max_f1`. Model B
requests may select `budget_0_10`, `budget_0_25`, `budget_0_50`, `budget_1_00`,
or `budget_2_00`; thresholds are recomputed from the existing validation split.
`GET /operating_points` reports validation volumes and historical test outcomes
including false positives. Model A supports only its existing maximum-F1
threshold. The test period is evaluated as recorded workload evidence only and
is not used to select thresholds.

The endpoint reads history from the fixed processed V1 reference dataset and
adds previously scored online events from a separate local SQLite store at
`models/online_history.sqlite`. An event is written only after feature
calculation, risk scoring, and explanation generation complete; the reference
Parquet files remain read-only. At startup, the service determines the maximum
step in the reference file; for the supplied local dataset this is step 743.
`POST /score_transaction` rejects requests with `step <= reference_max_step`
with HTTP 422, so mutable online events begin strictly after the immutable
baseline. Historical transactions continue to use `POST /score` unchanged.
Later online requests can use a saved event only when its step is strictly
earlier. Two transactions with the same step never use one another as history.
The SQLite history query also excludes any legacy stored online rows at or
before the reference maximum, preventing overlap with the baseline.

The optional `event_id` request field enables duplicate protection: a repeated
ID is rejected with HTTP 409. Without an ID, the API cannot safely infer whether
an identical-looking request is a retry or a distinct transaction, so each
successful request is recorded as a new event. The SQLite state survives API
restarts, but is local prototype state; the event simulator is a sequential
local live-stream demonstration, not a production streaming service. The API
returns feature, inference, explanation, and state-write timings and adds an
`X-Process-Time-Ms` response header; these diagnostics are not latency or
concurrency guarantees. Treat this as a transaction-time/live-stream
prototype, not a production service.

Illustrative response shape (numeric values vary with the trained artifacts and
persisted online history; the full response also includes remaining feature
contributions and the model input feature dictionary):

```json
{
  "model": "B",
  "operating_point": "max_f1",
  "risk": {
    "score": 98.59,
    "score_kind": "model score; not a calibrated probability",
    "calibrated_probability": null,
    "review_priority": "High review priority",
    "review_threshold": 97.69,
    "flagged_for_review": true
  },
  "evidence_strength": {
    "status": "New",
    "sender_prior_transactions": 0,
    "receiver_prior_transactions": 0,
    "established_history_minimum": 5
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
  "history_rule": "Only reference and previously scored online transactions with step strictly less than this event were used."
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
transaction-request validation, and dashboard-facing API routes. The existing
`tests/validate_output.py` is the separate validation tool for the original V1
preprocessing outputs.

## Scope and limitations

SyndicAI is an investigator-prioritization prototype evaluated on synthetic
PaySim data. It does not determine that a transaction is fraudulent and has no
fraud-recovery, compliance-certification, latency, or production-throughput
claim. Model B is the current default because behavioural features improved
measured held-out results over transaction-only Model A, while Model C tied
Model B and did not use its network features in the recorded run. NetworkX
context is for analyst investigation, not evidence that Model C improves risk
ranking. The anomaly detector is a rejected offline experiment and is not
loaded by production scoring.

Transaction-time state is stored in a local SQLite file under `models/`, which
is Git-ignored and tied to the local reference dataset/model artifacts. It is
not a shared or managed event store. `event_id` is optional; duplicate
protection applies only when callers supply it. Events sharing a step are not
ordered against one another. The online scorer supports Models A and B only;
new-event network scoring is not implemented.