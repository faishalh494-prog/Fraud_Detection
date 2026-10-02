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

## Model B: validation-selected alert-budget comparison

This controlled comparison reuses the existing trained Model B, the fixed
chronological validation split (980,416 rows; 564 frauds), and untouched test
split (918,617 rows; 4,006 frauds). Model weights, features, preprocessing, and
splits are unchanged. For each target burden, the threshold was selected only
from validation scores by finding the realizable score cutoff whose validation
alert count is closest to the target count (rounded to the nearest row). When
equally close, the cutoff at or below the budget is preferred. Repeated score
values can prevent exact target counts; the selected validation burden is
shown below. The existing validation maximum-F1 threshold is included as the
reference point and was not changed.

After choosing all cutoffs on validation, the existing Model B scored the test
split once; each row below reports the resulting confusion counts at its
validation-selected cutoff. Test metrics did not inform cutoff selection.
PR-AUC is average precision over the continuous Model B test scores, so it is
the same for every threshold.

| Operating point | Validation threshold | Validation alert burden | Test precision | Test recall | Test F1 | Test PR-AUC | Test alerts | False positives | False negatives | Test alert burden | False positives / 10,000 negatives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Target 0.10% | 0.965717 | 0.0983% | 0.4370 | 0.4988 | 0.4658 | 0.5093 | 4,572 | 2,574 | 2,008 | 0.4977% | 28.14 |
| Target 0.25% | 0.959042 | 0.3252% | 0.3607 | 0.5657 | 0.4405 | 0.5093 | 6,283 | 4,017 | 1,740 | 0.6840% | 43.92 |
| Target 0.50% | 0.941377 | 0.4988% | 0.2996 | 0.6308 | 0.4062 | 0.5093 | 8,435 | 5,908 | 1,479 | 0.9182% | 64.60 |
| Target 1.00% | 0.910207 | 1.0000% | 0.2169 | 0.7808 | 0.3395 | 0.5093 | 14,419 | 11,291 | 878 | 1.5696% | 123.45 |
| Target 2.00% | 0.596373 | 2.0000% | 0.1376 | 0.8465 | 0.2368 | 0.5093 | 24,635 | 21,244 | 615 | 2.6817% | 232.27 |
| Existing validation max-F1 | 0.976926 | 0.0337% | 0.6490 | 0.3984 | 0.4937 | 0.5093 | 2,459 | 863 | 2,410 | 0.2677% | 9.44 |

### Investigator workload trade-offs

The test alert burden is substantially higher than the corresponding validation
burden at every selected target. For example, a 0.10% validation budget became
0.4977% of test transactions, while 1.00% validation became 1.5696% on test.
These are temporal distribution changes, not post-hoc threshold adjustments;
the test set remains a holdout.

Relative to maximum F1, the 0.10% validation operating point raises test recall
from 39.84% to 49.88% and lowers missed frauds from 2,410 to 2,008, but increases
alerts from 2,459 to 4,572 and false positives from 863 to 2,574. The 0.50%
point finds 2,527 frauds (63.08% recall) with 8,435 alerts, of which 5,908 are
false positives. At 1.00%, recall reaches 78.08% and misses fall to 878, at a
cost of 14,419 test alerts and 11,291 false positives. The 2.00% point finds
the most frauds in this comparison (84.65% recall), but generates 24,635 alerts,
21,244 false positives, and 232.27 false positives per 10,000 legitimate test
transactions.

Maximum F1 remains the best F1 row in this comparison; the broader operating
points exchange precision and F1 for greater coverage. A nominal validation
alert budget is not a guaranteed test workload, so investigators should treat
the test burden shift as a material capacity risk. No alternative threshold is
recommended or deployed by this analysis: the appropriate workload/coverage
trade-off requires an explicit investigator capacity decision.

The focused selector and metric tests are in
[`tests/test_alert_budget_evaluation.py`](../tests/test_alert_budget_evaluation.py).
