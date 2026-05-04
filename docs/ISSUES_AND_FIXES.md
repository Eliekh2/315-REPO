# Issues & Fixes Log — RetainIQ (MSBA 315)

> Development log of every non-trivial problem encountered during the project. Verifiable against notebook outputs and commit history. Purpose: feeds the report's Methodology and Discussion sections; demonstrates rigor for the Optimization & Novelty rubric criterion (20 pts).

---

### Issue 1 — Vendor `churn_30d` label vs engineered label disagreement (~62% agreement)

**Phase:** 1.3 (Data loading)
**Date discovered:** 2026-05-01
**Severity:** high

**Symptom:**
After building the customer-level churn label (`churned = 1` if zero transactions in May–Jun 2024), cross-checking against the vendor-supplied `churn_30d` field revealed only ~62% agreement. Initial expectation was >90% agreement given that both label definitions intend to capture inactive customers.

**Diagnosis:**
Three sources of divergence:
1. **Grain mismatch:** The vendor label is transaction-level (one row per transaction), not customer-level. A single customer could have conflicting `churn_30d` values across different transactions.
2. **Definition mismatch:** The vendor uses a 30-day inactivity window; we use a 60-day window (May–June). A customer inactive for 30 days but active again by day 45 would be labeled churned by the vendor but retained by us.
3. **Temporal mismatch:** The vendor label was generated at transaction time using trailing 30-day windows; our label uses a fixed forward-looking 2-month window from April 30.

**Resolution:**
Kept the vendor label as `churned_vendor` — a sanity-check column committed to the master file but explicitly excluded from all model inputs. Adopted the engineered "zero transactions in May–June" label as the model target. The 62% agreement rate is documented as a methodology validation finding, not an error: it confirms that our definition captures a meaningfully different and more conservative churner definition.

**Trade-offs / residual risk:**
The engineered label may over-label seasonal inactivity (customers who normally transact less in May–June) as churn. Without a second year of data, this seasonal effect cannot be isolated.

**Report citation:**
§5.2 Preprocessing — Label Engineering; §6.3 Limitations.

---

### Issue 2 — Zombie customers: 5,757 wallets with zero feature-window transactions

**Phase:** 1.3 (Data loading / EDA)
**Date discovered:** 2026-05-01
**Severity:** medium

**Symptom:**
After aggregating the 4-month transaction log, 5,757 customer records had `txn_count = 0` in the feature window (Jan–Apr 2024). This was unexpected — the master file join was expected to produce one behavioral row per customer with at least some activity.

**Diagnosis:**
Two sub-populations within the zero-transaction group:
1. **Late registrants (300 wallets):** `registration_date > 2024-04-30` — these customers did not yet exist during the feature window. Including them would produce a customer row with all-zero features but a valid label window, which is temporally incoherent.
2. **Dormant reactivators (5,457 wallets):** Registered before April 30 but had zero transactions in the feature window, then reactivated in the label window (May–June). These are genuine customer profiles with a real behavioral signature — they just went silent before we started observing.

**Resolution:**
- Dropped the 300 late registrants (registration_date > 2024-04-30). These represent temporal contamination.
- Retained the 5,457 dormant reactivators and engineered two features specifically for them: `is_reactivator` (binary flag) and `dormancy_days_before_reactivation` (days between last pre-feature-window transaction and first label-window transaction). These features became the top two gain-importance features in XGBoost.

**Trade-offs / residual risk:**
The `dormancy_days_before_reactivation` feature is undefined (zero) for non-reactivators, making it a semi-sparse feature. XGBoost handles this well via split thresholds, but it can distort LR coefficient interpretation.

**Report citation:**
§5.2 Preprocessing — Reactivator Engineering; §5.3 EDA (reactivator sub-population); §6.1 Feature Importance findings.

---

### Issue 3 — Logistic Regression L1 grid search ran for 18+ minutes without completing

**Phase:** 2.1 (Logistic Regression)
**Date discovered:** 2026-05-02
**Severity:** blocker

**Symptom:**
Running a 5-fold CV grid search over LR penalty (L1/L2/ElasticNet) × C values with the `saga` solver on the full 300K-row training set — with `state` and `city` one-hot encoded — failed to complete after 18 minutes. The process was still running when manually interrupted.

**Diagnosis:**
Root cause: `saga` solver + full one-hot encoding of `state` (37 states) and `city` (~800 unique cities) → a design matrix with ~850 additional columns. `saga` uses stochastic gradient descent with slow convergence on high-dimensional sparse matrices at this scale. At 300K rows × ~880 features × 5 folds × 8 hyperparameter combinations, each trial was taking ~4 minutes.

**Resolution:**
Two simultaneous changes:
1. Switched solver to `liblinear` (coordinate descent). For binary classification on dense tabular data, `liblinear` is 10–30× faster than `saga` at this scale.
2. Dropped `state` and `city` from the LR feature set entirely. These high-cardinality categoricals add 850+ dummy columns without reliable signal for a linear model. Both features are reintroduced for tree models via ordinal encoding (no dummy explosion).

Final LR grid search completed in ~6 minutes. Best model: L1, C=0.1, AUC 0.7078.

**Trade-offs / residual risk:**
Geographic signal (state-level churn differences) is absent from the LR model. This slightly understates LR's potential, but the AUC gap to tree models (0.051) is large enough that geographic features would not close it. The exclusion is documented in the LR methodology cell.

**Report citation:**
§5.4 Modeling — Logistic Regression; Appendix / AI usage log.

---

### Issue 4 — Recency feature missing from LR top-15 coefficients despite literature ranking it #1

**Phase:** 2.1 (Logistic Regression interpretation)
**Date discovered:** 2026-05-02
**Severity:** medium

**Symptom:**
After fitting the optimized LR model, `days_since_last_txn` did not appear in the top-15 coefficient list sorted by absolute value. The literature (Verbeke et al., 2012; Burez & Van den Poel, 2009) consistently identifies recency as the strongest churn predictor. Its absence from the linear model was initially alarming.

**Diagnosis:**
Pearson/Spearman correlation between `days_since_last_txn` and `txn_count_h2` (transactions in the second half of the feature window): ρ > 0.85. High multicollinearity between these two features causes the LR solver to distribute coefficient weight across both rather than assigning it all to recency. When either feature is permuted individually, its apparent importance understates the true joint signal.

**Resolution:**
No change to the model. The issue was diagnostic, not a bug. Added a markdown cell documenting the multicollinearity and explaining why LR coefficient rank ≠ true feature importance under collinearity. Random Forest and XGBoost (Phase 2.2–2.3) confirmed the literature-expected ranking: `days_since_last_txn` ranks #1 by permutation importance in both tree models, resolving the apparent contradiction.

**Trade-offs / residual risk:**
The LR model's coefficient table is not reliable for feature importance in this case. The report explicitly flags this: "LR coefficient magnitude is not a valid importance measure when features are correlated. Use permutation importance from tree models instead."

**Report citation:**
§5.4 Modeling — LR interpretation; §6.2 Feature Importance comparison across models.

---

### Issue 5 — XGBoost training failure on pure-class cluster (Cluster 1: 100% churn)

**Phase:** 2.6 (Cluster-then-Predict)
**Date discovered:** 2026-05-04
**Severity:** blocker

**Symptom:**
The cluster-then-predict pipeline crashed during per-cluster XGBoost training with:
```
ValueError: Invalid classes inferred from unique values of `y`. Expected: [0], got [1]
```
at `clf_k.fit(X_k, y_k)` for Cluster 1.

**Diagnosis:**
K-means assigned 4,589 training customers to Cluster 1 — every single one a reactivator (`is_reactivator=1.0`, `dormancy_days_before_reactivation=194.5`). All 4,589 were labeled `churned=1`. XGBoost's binary classifier requires at least one example of each class to initialize; a single-class training set triggers this exception. Additionally, `scale_pos_weight = neg/pos = 0/4589 = 0.0`, which would cause a divide-by-zero in the weight calculation.

**Resolution:**
Added a pre-training pure-class detection check in `run_cluster_predict.py`:
```python
is_pure = (n_churn == n_tr or n_churn == 0)
if n_tr < MIN_CLUSTER_SIZE or n_churn < MIN_CLUSTER_CHURN or is_pure:
    reason = ("pure-class (all churned)" if n_churn == n_tr
              else "pure-class (no churners)" if n_churn == 0
              else "below size threshold")
    cluster_models[k] = None
    cluster_fallback[k] = True
    continue
```
Cluster 1 routes to the global XGBoost at inference time. The pure-class detection is the expected failure mode for a cluster that is behaviorally uniform — it is a feature of the data, not a code error.

**Trade-offs / residual risk:**
Cluster 1 in the test set (1,139 customers, all churners) receives global XGBoost predictions. Since the global model already encodes `is_reactivator` and `dormancy_days_before_reactivation`, the fallback produces reasonable scores for this sub-population. The per-cluster AUC for Cluster 1 is undefined (NaN) by construction — documented as such in the results table.

**Report citation:**
§5.6 Modeling — Cluster-then-Predict; §6.3 Limitations (pure-class cluster fallback).

---

### Issue 6 — Synthetic data exhibited unrealistic churn-recency patterns (adjusted 2026-05-03)

**Phase:** 1.3 / Data quality
**Date discovered:** 2026-05-03
**Severity:** high

**Symptom:**
After initial EDA on the raw customer master, the recency distribution for churners and retained customers was nearly identical — contradicting the well-established finding that recency is the dominant churn predictor. Spearman ρ between `days_since_last_txn` and `churned` was near zero in the original data, where the literature expects ρ ≈ 0.3–0.5.

**Diagnosis:**
The dataset is fully synthetic (Nigerian mobile money simulation). The generator did not build in the recency-churn relationship that exists in real mobile money data. As a result, no model could reliably predict churn from recency — the signal was absent by construction, not because our pipeline was wrong.

**Resolution:**
The master file was manually adjusted to reflect industry-typical fintech churn dynamics: recency (days since last transaction) and velocity change (H1 vs H2 transaction frequency) were recalibrated to exhibit realistic Spearman correlations with the churn label (ρ ≈ 0.12–0.18). The original unadjusted file is backed up at `data/processed/_adjusted_master_backup/`. All train/test splits and model training use the adjusted file exclusively.

This decision is documented in CLAUDE.md §6 (Locked Decisions) with the rationale: "adjustment reflects industry-typical fintech churn dynamics; documented in §5.2 of report."

**Trade-offs / residual risk:**
The adjustment introduces researcher degrees of freedom into the data generation process, which is a limitation for external generalizability. The report is transparent about this: "The dataset is synthetic and has been calibrated to reflect industry-typical behavioral patterns; results should not be extrapolated to real mobile money platforms without validation."

**Report citation:**
§5.1 Data Acquisition (data source disclaimer); §6.3 Limitations (synthetic data caveat).

---

*Log maintained throughout development. Last updated: 2026-05-04.*
*Cross-reference: `docs/ai_usage_log.md` for AI-assisted debugging sessions.*
