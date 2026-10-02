# Offline research: normal-behaviour anomaly

This is a rejected, offline experiment and is not part of the production
`src/syndicai_v4` package, inference API, or investigator dashboard.

Run it from the repository root after training the V4 baseline:

```powershell
python -m experiments.normal_behaviour_anomaly.anomaly_experiment --estimators 100 --folds 5
```

Run its focused tests separately from the production V4 tests:

```powershell
python -m unittest discover -s tests -p "test_normal_behaviour_anomaly.py"
```

It reuses `data/processed/syndicai_v1_{train,val,test}.parquet` and the existing
`models/model_bundle.joblib` / `models/metrics.json`. It writes experimental
models, scores, and metrics under the Git-ignored `models/anomaly_experiment/`
directory. It does not rebuild the V1 feature pipeline or modify A/B/C
artifacts.

## Method

Isolation Forest uses the eight existing prior-only behavioural features:
receiver lifetime count, prior amount sum, prior average amount, steps since
last receiver activity, receiver-new flag, last-24-step receiver count, sender
lifetime count, and sender-new flag. Skewed values are `log1p` transformed.
The no-history sentinel for steps-since-last is mapped to zero while the
receiver-new flag remains separate.

The detector uses 100 estimators, up to 512 samples per tree,
`contamination="auto"`, and a fixed seed. It is fit only on legitimate training
rows. Five expanding chronological blocks produce training scores using
legitimate examples from strictly earlier steps only. The initial warm-up
block (1,030,299 rows) is excluded from the augmented classifier and a matched
B-only control because no earlier history exists to score that block. The
remaining 3,433,288 training rows are used for that controlled comparison.
The original Model B remains the full-training product baseline.

Validation and test anomaly scores use the final detector fitted on legitimate
training-period rows. Training labels only exclude positive rows from fitting;
they are not detector features or a supervised target. Anomaly-only and
B-plus-anomaly thresholds are independently selected by maximum F1 on
validation. Test data is used only for final evaluation.

## Recorded results

These are measured on the supplied PaySim artifacts, not production
performance claims:

| Signal/model | Validation threshold | Test precision | Test recall | Test F1 | Test PR-AUC | Test false positives | Test alerts | Alert burden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing Model B | 0.976926 | 0.6490 | 0.3984 | 0.4937 | 0.5093 | 863 | 2,459 | 0.2677% |
| Anomaly only | 0.628525 | 0.0044 | 0.0130 | 0.0066 | 0.0049 | 11,806 | 11,858 | 1.2909% |
| B + anomaly feature | 0.974539 | 0.4776 | 0.4341 | 0.4548 | 0.4850 | 1,902 | 3,641 | 0.3964% |
| B-only matched-row control | 0.974567 | 0.4704 | 0.4319 | 0.4503 | 0.4812 | 1,948 | 3,678 | 0.4004% |

The augmented model is slightly better than its matched-row control
(PR-AUC +0.0039, F1 +0.0045, 46 fewer false positives), but worse than the
existing full-training Model B (PR-AUC -0.0242, F1 -0.0389, 1,039 more false
positives and 1,182 more alerts). The matched control itself is substantially
below full-training B, so the small gain does not justify integration. The
anomaly-only PR-AUC (0.0049) is near the test fraud prevalence (0.436%), with
very low precision and high alert burden.

**Decision:** do not promote anomaly scoring into the V4 risk engine or
investigator/API. Retain only as a reproducible offline experiment.

Detailed machine-readable metrics and research models are generated locally in
`models/anomaly_experiment/`; these ignored artifacts were not moved or
modified when separating the source code.
