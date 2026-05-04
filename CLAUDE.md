# CLAUDE.md — RetainIQ MSBA 315 Project

> **For Claude Code (and any AI assistant working in this repo):** Read this file at the start of every session. It contains the complete project context, the grading rubric, the locked plan, the data schema, and the rules of engagement. Do not skip it. Do not ask Elie to re-explain things that are written here.

---

## 1. Who You Are Working With

- **Name:** Elie
- **Location:** Jounieh, Lebanon
- **Role:** Solo developer on this project (originally a team of 6, but Elie is executing the full pipeline alone)
- **Course:** MSBA 315 — Machine Learning & Predictive Analytics, AUB
- **Instructor:** Wael Khreich (wk47@aub.edu.lb)
- **Technical level:** Learning developer; uses Claude as primary technical partner
- **Working style:** Direct. No fluff. Push back when something is wrong. Don't pad. Don't over-apologize. If a plan is bad, say so and propose an alternative. Don't ask permission for trivial details — make a sensible call and explain it briefly.

---

## 2. The Project — RetainIQ

**Public framing (for the report and presentation):** A machine learning system for predicting customer churn in mobile money / digital wallet platforms, built on the Nigerian mobile money market as a proxy for the broader MENA region.

**Real framing (the actual business goal):** RetainIQ is a B2B AI churn prediction and retention platform targeting Lebanese and MENA digital wallets. Priority targets are Whish Money (1.5M users, $5B/year, 90% Lebanon market share, CEO Toufic Koussa is an AUB graduate) and OMT Pay. The MSBA 315 deliverable is the ML proof-of-concept that becomes the demo for that pitch.

**Relationship to MSBA 305:** MSBA 305 was the data pipeline course project (submitted April 20, 2026; presented April 25, 2026). The 305 work is reusable as Section 5.1 (Data Acquisition) and 5.2 (Preprocessing) of the 315 report. This must be documented inside the 315 methodology, framed in ML terms — not skipped.

---

## 3. The Datasets (CONFIRMED SCHEMA)

Source files (read-only):
```
C:\Users\User\Desktop\315 - retain IQ\Datasets selected\
├── Customer Profiles JSON\customer_profiles.json
└── Nigerian datasets PARQUET\nigerian_mobile_money_full.parquet
```

### Parquet — Transaction log

- **Shape:** 4,000,000 rows × 13 columns
- **Unique customers:** 375,837
- **Date range:** January–June 2024 (6 months)
- **Nulls:** Zero across all columns
- **Vendor target label:** `churn_30d` — 240K True (6%), 3.76M False (94%) — **kept as sanity-check column only; not used as the model target**
- **Key columns:** `wallet_id`, `transaction_type` (7 categories), `channel` (4), `device_os` (3), `kyc_tier` (3), `amount_ngn`, `fee_ngn`, `balance_after_ngn`, `fraud_flag`, `agent_id`, plus a timestamp column.

### JSON — Customer profiles

- **Shape:** 375,837 records nested under `data[]`
- **Join key:** `wallet_id` (100% coverage with Parquet)
- **Fields:** `age`, `gender`, `state`, `city`, `registration_date`, `account_status` (active/dormant/suspended), `referral_source`, `preferred_language` (English/Hausa/Yoruba/Igbo/Pidgin), `linked_bank`, `support_tier` (standard/silver/gold)
- **Nulls:** Zero

---

## 4. The Rubric — Every Decision Must Be Grading-Aware

The report is graded out of 100 points (from `Rubric_project_report.pdf`):

| Section | Points | What it rewards |
|---|---|---|
| Abstract (300 words max) | 5 | Purpose, methodology, findings, conclusions |
| Introduction | 5 | Background, problem, proposed solution, research questions, contribution |
| Literature Review | 20 | Recent + relevant papers, comparison of methodologies, datasets, gaps, best results |
| **Methodology — Protocol** | **20** | Precise description of datasets, preprocessing, algorithms, training, evaluation, error analysis, metrics |
| **Methodology — Optimization & Novelty** | **20** | Optimization of preprocessing, features, hyperparameters; **what was done to outperform related work**; **novel approaches/ideas** |
| Results & Discussion | 20 | Charts, tables, error analysis, comparison with related work, research questions answered |
| Conclusions & Recommendations | 10 | Limitations, technical and business recommendations, future work |

**Course-level grade weights** (from `MSBA315ProjectDescription.pdf`):
- Report: **50%**
- Presentation (10–15 slides): **25%**
- Implementation (notebook clarity, structure, comments, replicability): **25%**

**Implication for every line of code:**
- Methodology = 40 points combined. Most heavily weighted block. Every choice must be defensible and documented in markdown cells.
- "Novel approaches" is explicitly named and worth real points. A vanilla LR/RF/XGBoost benchmark caps in the B range without a novelty layer.
- Comparison with related work is required. Since the data is not a published benchmark, frame the contribution as: deployability + interpretability + multi-source integration + a novel two-stage architecture (cluster-then-predict).

---

## 5. The Locked Plan (Solo Execution)

Phases run **sequentially**, not in parallel. One person, one critical path.

### Phase 1 — Foundation (build once, never revisit)

| Step | Output |
|---|---|
| 1.1 — Data inspection (already done, schema confirmed in §3) | — |
| 1.2 — Implement `src/data_loader.py` | runnable module |
| 1.3 — Run loader → produce `data/processed/customers_master.parquet` | 375K rows × ~45 cols |
| 1.4 — EDA notebook | `notebooks/01_eda.ipynb` + figures |
| 1.5 — Stratified 80/20 train/test split | `train.parquet`, `test.parquet` (frozen) |
| 1.6 — Implement canonical `src/evaluate.py` | one function used by every model |

### Phase 2 — Modeling Tracks (sequential, one notebook at a time)

| # | Notebook | What it covers |
|---|---|---|
| 2.1 | `02_logistic_baseline.ipynb` | LR with L1/L2/ElasticNet, scaling, RFE feature selection, calibration. **The required interpretable baseline.** |
| 2.2 | `03_random_forest.ipynb` | RF with hyperparameter search, OOB error, permutation importance. **The bagging benchmark.** |
| 2.3 | `04_xgboost.ipynb` | Deep XGBoost tuning (Optuna), early stopping, regularization. **Primary model.** |
| 2.4 | `05_lightgbm_catboost.ipynb` | LightGBM + CatBoost vs XGBoost head-to-head. **Boosting alternatives.** |
| 2.5 | `06_imbalance_study.ipynb` | No treatment vs SMOTE vs SMOTEENN vs class_weight vs scale_pos_weight vs threshold tuning vs cost-sensitive. **Standalone methodology contribution.** |
| 2.6 | `07_cluster_then_predict.ipynb` | K-means or DBSCAN on customer behavior → segment-specific XGBoost models. **The novelty layer. The A+ differentiator.** |

### Phase 3 — Synthesis & Deliverables

| Step | Output |
|---|---|
| 3.1 — SHAP analysis on winning model | `notebooks/08_shap_analysis.ipynb` + figures |
| 3.2 — Master end-to-end notebook | `notebooks/00_master_submission.ipynb` (the file actually submitted) |
| 3.3 — Research paper (5–10 pages) | `docs/research_paper.md` → exported PDF |
| 3.4 — Slides (10–15) | `docs/presentation.pptx` |
| 3.5 — AI usage log (course requirement) | `docs/ai_usage_log.md` |
| 3.6 — Final review against rubric | self-audit, gap-fix |

---

## 6. LOCKED DECISIONS — Do Not Re-Open These

These have been debated and resolved. Do not reopen unless Elie explicitly says so.

| Decision | Locked value | Why |
|---|---|---|
| Customer-level grain | One row per `wallet_id` (375K rows total) | Prediction unit = customer. Transaction grain causes label leakage in train/test split. |
| File join order | Aggregate Parquet first, join JSON profiles last | Avoids 4M-row profile duplication; matches inference flow (identity → behavior). |
| Feature window | **2024-01-01 → 2024-04-30** (4 months) | Source for all behavioral features. |
| Label window | **2024-05-01 → 2024-06-30** (2 months) | Source for the churn label only. **Never used to compute features.** |
| Churn label | `churned = 1` if zero transactions in label window, else `0` | Engineered from raw activity. Simple, defensible, no leakage. |
| Vendor `churn_30d` | Kept as sanity-check column `churned_vendor`. **Not the model target.** | Lets the report report agreement rate as a methodology validation. |
| Random seed | 42 everywhere | Reproducibility. |
| Train/test split | Stratified 80/20 on `churned` | Preserves class ratio. |
| Cross-validation | 5-fold stratified, on training set only | Standard. Test set never enters CV. |
| Test set policy | **Frozen after Phase 1.** No re-splits. No peeking. | Required for valid generalization estimate. |
| Path management | All paths come from `src/config.py`. **Never hardcode in notebooks.** | Single source of truth; portable. |
| LR feature exclusions | Drop `state`, `city`, `is_reactivator`, `dormancy_days_before_reactivation` from LR feature set | High-cardinality categoricals destabilize `liblinear`; system-derived flags inflate linear baseline; all reintroduced for tree models. `account_status` removed globally via data audit below. |
| LR solver | `liblinear` (L1 + L2 only) | `saga` + high-cardinality OHE + 300K rows is intractable; `liblinear` coordinate descent is 10–30× faster for binary classification at this scale |
| Feature audit (post-Phase 2.1) | Drop 7 features from master: `date_of_birth`, `registration_date`, `account_status`, `notification_preferences`, `last_txn_date`, `first_txn_date`, `support_tier` | JSON-synthesized fields (`account_status`, `notification_preferences`, `support_tier`) generated independently of Parquet transaction history — risk of label-contradicting signal. Raw-date fields (`date_of_birth`, `registration_date`, `last_txn_date`, `first_txn_date`) redundant with engineered derivatives (`age`, `tenure_days`, `days_since_last_txn`, `account_active_span_days`). |
| Metadata columns | Keep `wallet_id`, `full_name`, `churned_vendor` in master file but exclude from ALL model inputs | `wallet_id` = join key; `full_name` = output identifier for deployed demo; `churned_vendor` = sanity-check column. Never used as predictors. |
| OOB AUC computation | Use `oob_decision_function_[:, 1]` + `roc_auc_score`, NOT `oob_score_` (which returns accuracy, misleading for imbalanced data). | Accuracy is irrelevant for the imbalanced target; AUC is the canonical metric. |
| Adjusted master file | `data/processed/customers_master.parquet` was manually adjusted on 2026-05-03 to refine unrealistic patterns in synthetic data; loader marked as superseded; backup at `data/processed/_adjusted_master_backup/` | Original synthetic data exhibited unrealistic recency-churn patterns and insufficient behavioral heterogeneity; adjustment reflects industry-typical fintech churn dynamics; documented in §5.2 of report |

---

## 7. Repository Structure (use exactly this)

```
315 - retain IQ/                          ← parent folder Elie already created
│
├── Datasets selected/                    ← READ-ONLY source data, do not modify
│   ├── Customer Profiles JSON\
│   └── Nigerian datasets PARQUET\
│
└── retainiq/                             ← THIS is the VS Code repo root
    ├── CLAUDE.md                         ← this file
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    │
    ├── data/
    │   ├── interim/                      (intermediate cleaned files)
    │   └── processed/
    │       ├── customers_master.parquet        ← adjusted (375,537 × 38, canonical)
    │       ├── train.parquet                   ← adjusted train (300,429 × 38, FROZEN)
    │       ├── test.parquet                    ← adjusted test (75,108 × 38, FROZEN)
    │       └── _adjusted_master_backup/        ← backup of adjusted files (DO NOT DELETE)
    │
    ├── notebooks/
    │   ├── 00_master_submission.ipynb    (final, end-to-end, built last)
    │   ├── 01_eda.ipynb
    │   ├── 02_logistic_baseline.ipynb
    │   ├── 03_random_forest.ipynb
    │   ├── 04_xgboost.ipynb
    │   ├── 05_lightgbm_catboost.ipynb
    │   ├── 06_imbalance_study.ipynb
    │   ├── 07_cluster_then_predict.ipynb
    │   └── 08_shap_analysis.ipynb
    │
    ├── src/
    │   ├── __init__.py
    │   ├── config.py                     (paths, seeds, constants)
    │   ├── data_loader.py                (load + aggregate + label + join)
    │   ├── features.py                   (feature engineering helpers)
    │   ├── evaluate.py                   (canonical evaluation function)
    │   └── train_utils.py                (CV, splitting helpers)
    │
    ├── outputs/
    │   ├── figures/                      (all plots saved here, dpi=150)
    │   ├── models/                       (.pkl files, one per track)
    │   └── tables/
    │       └── benchmark.csv             (one row per model, appended)
    │
    └── docs/
        ├── research_paper.md
        ├── presentation.pptx
        └── ai_usage_log.md
```

---

## 8. Tech Stack (locked)

```
Python 3.11
pandas, numpy, pyarrow
scikit-learn
xgboost, lightgbm, catboost
imbalanced-learn (SMOTE family)
shap
optuna (hyperparameter tuning)
matplotlib, seaborn, plotly
jupyter, ipykernel
```

See `requirements.txt`.

---

## 9. Conventions (follow always)

**Code:**
- Type hints on all functions in `src/`.
- Docstrings on all public functions (Google style).
- All paths come from `src/config.py`. Never hardcode.
- All randomness uses `RANDOM_SEED = 42` from `config.py`.

**Notebooks:**
- First cell: title, purpose, last-updated date.
- Imports in cell 2.
- Markdown cells between code blocks explaining what and why.
- Save figures with `plt.savefig(FIGURES / "name.png", dpi=150, bbox_inches="tight")`.
- No leftover `print` debugging in the final version.
- Restart kernel + run all before considering a notebook done.

**Data hygiene:**
- Test set is FROZEN after Phase 1.
- Every transformation that uses statistics (mean, std, encoded categories, SMOTE) is fit on training data only.
- Use `sklearn.pipeline.Pipeline` to enforce this. Don't transform manually in notebooks.
- Append one row to `outputs/tables/benchmark.csv` after each modeling notebook.

**Reporting:**
- Every claim in the report ties back to a number in `benchmark.csv` or a figure in `outputs/figures/`.
- No metric appears in the paper that wasn't produced by `src/evaluate.py`.
- Literature review citations are real and verifiable. Never invent papers.

---

## 10. What Claude Code Should DO

- Read this file at the start of every session before suggesting any change.
- Before writing data code, check `src/config.py` and `src/data_loader.py` for existing conventions.
- Use the canonical `evaluate()` function — don't invent new metrics inline.
- Push back when a request would violate the plan (e.g., re-splitting test data).
- Add a markdown rationale cell when making non-obvious choices (hyperparameter range, feature transformation, threshold).
- Suggest the rubric tie-in when relevant ("this maps to the Optimization & Novelty section").
- Be proactive about fixing latent issues you spot (missing seed, leaking transformer, hardcoded path).

## 11. What Claude Code NEVER Does

- Never modify files in `Datasets selected/`.
- Never re-split train/test after Phase 1 freeze.
- Never apply SMOTE / scaling / encoding fits to the test set.
- Never use the test set inside cross-validation loops.
- Never hardcode file paths.
- Never commit raw transaction data.
- Never invent literature.
- Never claim "state of the art" without a published baseline to compare against.
- Never skip the markdown rationale for a non-trivial decision.

---

## 12. Bootstrap Sequence — First Session in VS Code

When opened fresh, run these in order:

```powershell
# 1. Open VS Code in the parent folder so it sees both Datasets and retainiq
cd "C:\Users\User\Desktop\315 - retain IQ"
code .

# 2. From VS Code's integrated terminal, set up the Python env
cd retainiq
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the data loader (Phase 1, Step 1.3) — first time takes ~2 minutes on full data
python -m src.data_loader

# 4. Verify output
python -c "import pandas as pd; df = pd.read_parquet('data/processed/customers_master.parquet'); print(df.shape); print(df['churned'].value_counts(normalize=True))"

# 5. Launch Jupyter for the EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

Expected output of step 4:
- Shape: roughly `(375837, ~45)`
- Churned distribution: roughly 60–80% retained / 20–40% churned (will know exactly after running)

---

## 13. Status Tracker

Keep this updated as work progresses. Claude Code should update it when a phase step finishes.

- [x] Proposal submitted (Feb 26, 2026)
- [x] Literature review submitted (Mar 25, 2026)
- [x] Data inspection complete (May 1, 2026)
- [x] Schema confirmed, plan locked (May 1, 2026)
- [x] Phase 1.2 — `src/data_loader.py` implemented
- [x] Phase 1.3 — `customers_master.parquet` generated
- [x] Phase 1.4 — EDA notebook complete
- [x] Phase 1.5 — train/test split frozen
- [x] Phase 1.6 — `src/evaluate.py` implemented
- [x] Phase 2.1 — Logistic regression baseline (re-run on adjusted data 2026-05-03; LR optimized AUC ~0.71)
- [x] Phase 2.2 — Random Forest (re-run on adjusted data 2026-05-03; RF tuned AUC ~0.758)
- [x] Phase 2.3 — XGBoost (2026-05-04; Vanilla AUC 0.7591, Tuned AUC 0.7590; Optuna 30 trials best CV AUC 0.7614; top perm feature: days_since_last_txn)
- [ ] Phase 2.4 — LightGBM + CatBoost
- [ ] Phase 2.5 — Imbalance study
- [x] Phase 2.6 — Cluster-then-predict (2026-05-04; K=3, ClusterPredict AUC 0.7585 vs global 0.7590; ceiling NOT broken; Cluster 1 = 100% churn reactivators — automatic trigger rule)
- [ ] Phase 3.1 — SHAP analysis
- [x] Phase 3.2 — Master notebook (2026-05-04; 00_master_submission.ipynb 42 cells Colab-ready; ISSUES_AND_FIXES.md 6 entries; PROJECT_STORY.md 4,362 words; pushed to https://github.com/Eliekh2/315-REPO)
- [ ] Phase 3.3 — Research paper
- [ ] Phase 3.4 — Slides
- [ ] Phase 3.5 — AI usage log
- [ ] Phase 3.6 — Final rubric audit

---

## 14. Communication Style With Elie

- Be direct.
- Don't pad. No "great question" openers.
- When proposing options, rank them with trade-offs.
- When uncertain, say "I don't know — here's how we find out" rather than guessing.
- Don't ask Elie to make trivial decisions. Make a defensible call, document it, move on.
- Time matters: hard deadline + a real Whish demo at the end.

---

## 15. Reference: Files in the Original Project

These four PDFs are stored in the `claude.ai` parent project (not in this repo, but referenced):
- `MSBA315ProjectDescription.pdf` — official assignment
- `Rubric_project_report.pdf` — grading rubric (already summarized in §4)
- `RETAINIQ_315_PROJECT_CONTEXT.pdf` — full business context
- `MSBA315_complete_context.pdf` — all 11 lecture transcripts + 3 past Claude conversation summaries

If Claude Code needs the lecture content (e.g., to justify a methodology choice from a specific lecture), ask Elie to paste the relevant section. Do not invent lecture content.

---

**End of CLAUDE.md. This file is the single source of truth for the project.**
*Last updated: May 3, 2026 (v2 sprint removed; adjusted master canonical; Phase 1.5/2.1/2.2 re-run on adjusted data; LR ~0.71 AUC, RF tuned ~0.758 AUC)*
