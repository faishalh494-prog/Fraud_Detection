# SyndicAI V4: measured model comparison

This report records the run produced by `python -m src.syndicai_v4.train
--estimators 80` on the locally supplied, validated V1 Parquet artifacts.
The run uses 4,463,587 training rows, 980,416 validation rows, and 918,617
test rows. It preserves the existing chronological split and does not rebuild
or resample the data. The training split contains 3,643 positives, validation
564, and test 4,006.

All models use the same 80-tree XGBoost configuration, training data, random
seed, and imbalance weight. Each model's threshold is selected for maximum F1
on validation; the test split is evaluated once at that selected threshold.
PR-AUC is average precision over test scores. Alert burden is the fraction of
test transactions at or above the review threshold. False positives per 10,000
negatives uses the actual number of non-fraud test rows as its denominator.

| Model | Inputs | Validation threshold | Validation F1 | Test precision | Test recall | Test F1 | Test PR-AUC | Test alerts | Test false positives | Alert burden |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | Transaction | 0.972051 | 0.4058 | 0.3649 | 0.4101 | 0.3862 | 0.3171 | 4,503 | 2,860 | 0.4902% |
| B | Transaction + behaviour | 0.976926 | 0.4541 | 0.6490 | 0.3984 | 0.4937 | 0.5093 | 2,459 | 863 | 0.2677% |
| C | Transaction + behaviour + network | 0.976926 | 0.4541 | 0.6490 | 0.3984 | 0.4937 | 0.5093 | 2,459 | 863 | 0.2677% |

Model B improves over A on test precision, F1, PR-AUC, and alert burden, while
its recall is slightly lower. Model C is exactly tied with B on the reported
test and validation metrics in this run. This is not evidence that the network
features improve prediction.

The four prior-step network features were computed and included in Model C.
Their prevalence in the full reference data ranges from 0.0069% to 0.1036% of
rows. None was used in a Model C tree split (zero XGBoost gain importance).
Therefore the measured evidence supports Model B as the simpler default for
risk ranking; Model C remains available for comparison, and the NetworkX
investigation context remains available in the case view.

At the selected thresholds, Model B flags 2,459 of 918,617 test transactions
and misses 2,410 of 4,006 labeled fraud examples. This is a prioritization
aid, not a complete detector or a fraud verdict. The source is synthetic
PaySim data; these values are not production performance claims and may not
generalize to real transaction streams. `models/metrics.json` contains the
complete measured report, including confusion counts, class counts, model
configuration, network feature prevalence, and feature importances.

## Normal-behaviour anomaly experiment

The separate controlled run was produced by
`python -m src.syndicai_v4.anomaly_experiment --estimators 100 --folds 5`.
It reused the existing V1 train/validation/test artifacts and did not modify
the A/B/C models or preprocessing. The anomaly input is the eight existing
prior-only behavioural features: receiver lifetime count, prior amount sum,
prior average amount, steps since last receiver activity, receiver-new flag,
last-24-step receiver count, sender lifetime count, and sender-new flag.
Skewed counts/amounts were `log1p` transformed; the `-1` new-history sentinel
for steps-since-last was mapped to zero while the receiver-new flag remains
separate.

Isolation Forest used 100 estimators, at most 512 samples per tree,
`contamination="auto"`, and a fixed seed. It was fitted only on legitimate
training rows. Five expanding chronological blocks produced training anomaly
scores: each block's detector used legitimate examples with strictly earlier
steps. The initial warm-up block (1,030,299 rows) was not used to train either
the augmented classifier or its matched B-only control, because no earlier
training history exists for those anomaly scores. This leaves 3,433,288 rows
for that controlled fit; the original full-training Model B remains the
primary product baseline. Validation/test anomaly scores use the final
detector fitted on all legitimate training-period rows. Training labels were
used only to exclude positive rows from each Isolation Forest fit; labels were
not detector inputs or an optimization target. Validation labels were used
only to select F1-maximizing thresholds.

| Signal/model | Validation threshold | Test precision | Test recall | Test F1 | Test PR-AUC | Test false positives | Test alerts | Alert burden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing Model B | 0.976926 | 0.6490 | 0.3984 | 0.4937 | 0.5093 | 863 | 2,459 | 0.2677% |
| Anomaly only | 0.628525 | 0.0044 | 0.0130 | 0.0066 | 0.0049 | 11,806 | 11,858 | 1.2909% |
| B + anomaly feature | 0.974539 | 0.4776 | 0.4341 | 0.4548 | 0.4850 | 1,902 | 3,641 | 0.3964% |
| B-only matched-row control | 0.974567 | 0.4704 | 0.4319 | 0.4503 | 0.4812 | 1,948 | 3,678 | 0.4004% |

The augmented model is slightly better than its matched-row control
(PR-AUC +0.0039, F1 +0.0045, 46 fewer false positives), but it is worse than
the actual full-training Model B (PR-AUC -0.0242, F1 -0.0389, 1,039 more false
positives and 1,182 more alerts). The matched control itself is substantially
below full-training B, so the small matched-row gain is not enough evidence to
displace the existing baseline. The anomaly-only PR-AUC (0.0049) is near the
test fraud prevalence (0.436%), with very low precision and high alert burden.

**Decision:** do not promote the anomaly score into the supervised risk engine
or expose it in the investigator/API view. Keep this run as an offline
experimental signal. The results are in the ignored local artifact
`models/anomaly_experiment/metrics.json`; the detector and augmented model are
also saved there for reproducibility, not for API serving.
