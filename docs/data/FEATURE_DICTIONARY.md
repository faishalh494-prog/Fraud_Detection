# SyndicAI V1 — Feature Dictionary

Applies to `syndicai_v1_processed.parquet` (and the `_train`/`_val`/`_test` splits,
which are identical in schema). 6,362,620 rows × 34 columns.

## 1. Raw columns (unchanged from PaySim, always present)

| Column | Type | Description |
|---|---|---|
| `step` | int32 | Simulation hour, 1–743 (~31 days). Data is chronologically sorted by this column. |
| `type` | category | Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER. |
| `amount` | float64 | Transaction amount (local currency units). |
| `nameOrig` | string | Sender account ID. "C" prefix = customer. 99.85% of values appear exactly once in the whole file. |
| `oldbalanceOrg` | float64 | Sender's account balance before the transaction. **Simulator artifact risk — see §4.** |
| `newbalanceOrig` | float64 | Sender's account balance after the transaction. **Simulator artifact risk — see §4.** |
| `nameDest` | string | Receiver account ID. "C" = customer, "M" = merchant. |
| `oldbalanceDest` | float64 | Receiver's balance before the transaction. Always 0 when `nameDest` is a merchant (not tracked). **Simulator artifact risk — see §4.** |
| `newbalanceDest` | float64 | Receiver's balance after the transaction. Same merchant caveat. **Simulator artifact risk — see §4.** |
| `isFraud` | int8 | **Target.** 1 = fraudulent. Occurs only in TRANSFER/CASH_OUT. |
| `isFlaggedFraud` | int8 | PaySim's own naive large-transfer flag. 16 positive rows total, all true fraud, ~0.2% recall — not a useful feature on its own. |

## 2. Model A — transaction-level features (row-independent, zero leakage risk)

| Column | Type | Description |
|---|---|---|
| `hour_of_day` | int16 | `step % 24`. Hour within the simulated day (0–23). |
| `day_of_sim` | int16 | `step // 24 + 1`. Day number of the simulation (1–32). |
| `is_night` | int8 | 1 if `hour_of_day` in [0,5]. Off-hours heuristic. Tested empirically; not a strong signal in this synthetic dataset, kept for real-world applicability. |
| `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER` | int8 | One-hot flags for `type`, provided alongside the categorical column so both tree-based and linear/sklearn models can consume it directly. |
| `amount_log1p` | float32 | `log1p(amount)`. Amount is heavily right-skewed (mean ≈180k, max ≈92M); log-transform for models sensitive to scale. |

`MODEL_A_FEATURES` in the pipeline script = `step, hour_of_day, day_of_sim, is_night, type_* (5), amount, amount_log1p`.

## 3. Model B — behavioural features (Model A + the features below)

All of these use **only transactions with `step` strictly less than the current row's `step`.** Transactions sharing the current row's step are never used, because `step` is an hour-bucket (avg. ~8,563 txns/step, max 51,352) with no reliable sub-hour ordering.

| Column | Type | Description |
|---|---|---|
| `receiver_txn_count_before` | int32 | How many prior transactions this `nameDest` has received. |
| `receiver_total_amount_before` | float32 | Sum of amounts received by this `nameDest` before now. |
| `receiver_avg_amount_before` | float32 | `receiver_total_amount_before / receiver_txn_count_before`, 0 if no prior history. |
| `receiver_steps_since_last` | int32 | Steps since this `nameDest` last received anything. **-1 sentinel** = no prior transaction. |
| `receiver_is_new` | int8 | 1 if this is the first time this `nameDest` has ever received a transaction. |
| `receiver_txn_count_last24_before` | int32 | Transactions to this `nameDest` in the trailing 24 steps (~1 day) before now — a burst/velocity signal distinct from lifetime count. |
| `sender_txn_count_before` | int32 | How many prior transactions this `nameOrig` has sent. Near-constant (99.85% of rows = 0) — see §5. |
| `sender_is_new` | int8 | 1 if this is this `nameOrig`'s first-ever transaction. |

`MODEL_B_FEATURES` = `MODEL_A_FEATURES` + the 8 columns above.

## 4. Diagnostic / high-leakage-risk group (kept in the file, EXCLUDED from default training lists)

These are computed for transparency and ablation studies only. Do not add them to a model's feature list without deliberately re-reading the rationale below and accepting the risk.

| Column | Type | Description | Why it's flagged |
|---|---|---|---|
| `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` | float64 | Raw balances (also raw columns, §1) | See below |
| `diagnostic_amt_to_oldbalanceOrg_ratio` | float32 | `amount / (oldbalanceOrg + 1)` | `amount == oldbalanceOrg` (ratio ≈ 1) occurs in **0.00%** of legitimate TRANSFER/CASH_OUT rows vs **97.82%** of fraud rows in the same subset. This is PaySim's own fraud-injection rule (fraudulent agents transfer the full balance), not a discovered behavioural pattern — a model trained on this is decoding the simulator, not learning fraud. |
| `diagnostic_orig_balance_delta` | float32 | `oldbalanceOrg - amount - newbalanceOrig` | Diagnostic view of the same artifact. |
| `diagnostic_dest_balance_delta` | float32 | `newbalanceDest - oldbalanceDest - amount` (customer dest only, NaN for merchant dest) | Related artifact on the receiving side. |
| `diagnostic_orig_fully_drained` | int8 | 1 if `newbalanceOrig == 0` and `oldbalanceOrg > 0` | Fraud rate **0.0042%** (flag=0) vs **0.5269%** (flag=1) — 125x lift, same root cause as above. |
| `diagnostic_dest_zero_zero_balance` | int8 | 1 if customer-dest `oldbalanceDest == newbalanceDest == 0` | Fraud rate **0.0668%** (flag=0) vs **2.4586%** (flag=1). Merchant balances are *always* zero (not tracked by the simulator); this flag catches the same "balance not populated" artifact leaking into customer-to-customer transfers specifically for fraud rows. |

**Recommendation:** if the modelling team wants to test these, do it as an explicit ablation (train with vs. without) and treat a large jump in AUC/recall from adding them as a red flag that the model is exploiting a simulator quirk, not learning generalizable behaviour — not as a validation that the features are good.

## 5. Features considered and deliberately NOT built

| Candidate | Why not |
|---|---|
| `receiver_unique_senders_before` | Checked directly: **every single (`nameDest`,`nameOrig`) pair in the entire 6,362,620-row file is unique** — no sender ever sends to the same receiver twice, anywhere. This makes the feature mathematically identical to `receiver_txn_count_before` for every row (zero new information), while being the single most expensive computation in the whole pipeline (its underlying groupby doesn't compress the data at all). Dropped. |
| `sender_total_amount_before`, `sender_avg_amount_before`, `sender_median_amount_before` | 99.85% of `nameOrig` values appear exactly once in the whole file; of the 8,213 fraud transactions, only 28 come from a sender who transacts more than once *anywhere*. Lifetime average/median over 0–2 points is mostly undefined or meaningless. |
| `sender_unique_previous_beneficiaries` | Same reason — senders essentially don't have a "previous" to speak of. |
| Median previous amount (receiver side) | Adds real computational cost (no simple vectorized running-median) for marginal value over the mean, given typical prior-transaction counts. |
| Global (dataset-wide) transaction velocity | Reflects simulator throughput, not any specific account's behaviour; not a meaningful fraud signal here. |

## 6. Other columns

| Column | Type | Description |
|---|---|---|
| `split` | string | `"train"` / `"val"` / `"test"`, assigned by `step` (chronological, see report). |

## 7. Loading a model's inputs

```python
from syndicai_preprocessing import load_model_inputs
import pandas as pd

df = pd.read_parquet("syndicai_v1_processed.parquet")
X_train_A, y_train_A = load_model_inputs(df, model="A", split="train")
X_train_B, y_train_B = load_model_inputs(df, model="B", split="train")
```
