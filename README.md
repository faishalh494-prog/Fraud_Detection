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

### Optional normal-behaviour anomaly experiment

After training the V4 baseline, run the separate experiment:

```powershell
python -m src.syndicai_v4.anomaly_experiment
```

This fits Isolation Forest models only on legitimate rows from strictly earlier
training steps, chooses anomaly and augmented-model thresholds on validation,
and evaluates once on test. It writes results under the ignored
`models/anomaly_experiment/` directory. It is intentionally not part of the
inference API unless the measured experiment justifies promotion.

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

The `POST /score` endpoint scores a zero-based row from the validated processed
reference dataset. This version does not claim to compute leakage-safe historical
features for arbitrary live events outside that reference data.

## Tests

```powershell
python -m unittest discover -s tests -p "test_syndicai_v4.py"
```

Tests cover time-causal graph features, model threshold/evaluation helpers,
TreeSHAP explanations, and persisted investigation states. The existing
`tests/validate_output.py` is the separate validation tool for the original V1
preprocessing outputs.