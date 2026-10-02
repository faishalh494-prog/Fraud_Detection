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

The rejected normal-behaviour anomaly study is an offline experiment, separate
from production V4. Its methodology and measured results are documented in
[experiments/normal_behaviour_anomaly/README.md](../experiments/normal_behaviour_anomaly/README.md).
