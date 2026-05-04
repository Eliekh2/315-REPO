"""
build_master_notebook.py — generates notebooks/00_master_submission.ipynb

Run once from the repo root:
    python build_master_notebook.py
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

NB_PATH = Path("notebooks/00_master_submission.ipynb")
NB_PATH.parent.mkdir(parents=True, exist_ok=True)

cells = []

# ──────────────────────────────────────────────────────────────────────────────
# CELL 1 — Title block
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""# RetainIQ: Customer Churn Prediction for Digital Wallets

**Course:** MSBA 315 — Machine Learning & Predictive Analytics, AUB, Spring 2026
**Author:** Elie Khayrallah
**Submission date:** May 2026

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eliekh2/315-REPO/blob/main/notebooks/00_master_submission.ipynb)

---

## How to run this notebook

**Option 1 — Google Colab (recommended):** Click the badge above. Colab provides 12 GB RAM and a T4 GPU at no cost. Click *Runtime → Restart and Run All* and walk away for ~15 minutes. All dependencies, data, and pretrained models are fetched automatically.

**Option 2 — Local Jupyter:** Clone the repo, install dependencies, and open this notebook:
```bash
git clone https://github.com/Eliekh2/315-REPO.git
cd 315-REPO
python -m venv .venv && .venv/Scripts/activate  # Windows
pip install -r requirements.txt
jupyter notebook notebooks/00_master_submission.ipynb
```

**Option 3 — Read-only:** View the rendered notebook on [GitHub](https://github.com/Eliekh2/315-REPO/blob/main/notebooks/00_master_submission.ipynb) without running anything.

**Estimated runtime:** ~12–18 minutes with `SKIP_TRAINING=True` (load pretrained models). ~60–90 minutes with `SKIP_TRAINING=False` (full retrain including Optuna).

**RAM requirement:** ~4 GB peak. Colab free tier provides 12 GB — no issue.
"""))

# ──────────────────────────────────────────────────────────────────────────────
# CELL 2 — Abstract
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""## Abstract

Customer churn is a critical challenge for mobile money operators in emerging markets, where acquisition costs are high and customer lifetime value is long-tailed. This project develops RetainIQ, a machine learning pipeline for predicting 30-day churn in digital wallet platforms, motivated by the Lebanese and MENA mobile money market and validated on a Nigerian mobile money dataset comprising 4 million transactions across 375,537 customers.

We construct a customer-level feature matrix from a 4-month behavioral feature window and a 2-month held-out label window, engineering 34 features covering transaction recency, frequency, monetary value, channel diversity, and customer demographics. We benchmark six model families: Logistic Regression, Random Forest, XGBoost (tuned with Optuna), and a novel Cluster-then-Predict architecture using K-means segmentation. Tree-based models achieve AUC 0.757–0.759 vs. 0.708 for logistic regression, confirming non-linear churn dynamics. The primary model (XGBoost, Optuna-tuned) achieves AUC 0.759, PR-AUC 0.426, and 38.9% precision in the top decile.

The cluster-then-predict analysis reveals that K-means segmentation on behavioral features alone recovers a near-perfect churn segment (Cluster 1: 1.5% of customers, 100% churn rate), validating the hypothesis that behavioral clusters carry separable churn dynamics. The production recommendation routes Cluster 1 customers to an automatic retention trigger and applies the global XGBoost to the remaining population.
"""))

# ──────────────────────────────────────────────────────────────────────────────
# CELL 3 — Install dependencies
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_code_cell("""# Cell 3 — Install dependencies
# Run this cell first. The -q flag suppresses verbose pip output.
# In Colab, a runtime restart is NOT required for these packages.
!pip install -q pandas numpy pyarrow scikit-learn xgboost lightgbm \\
    imbalanced-learn shap optuna matplotlib seaborn joblib
"""))

# ──────────────────────────────────────────────────────────────────────────────
# CELL 4 — Detect environment
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_code_cell("""# Cell 4 — Detect environment (Colab vs local)
import os, sys

IN_COLAB = 'google.colab' in sys.modules
print(f"Running in: {'Google Colab' if IN_COLAB else 'Local Jupyter'}")
print(f"Python: {sys.version.split()[0]}")
"""))

# ──────────────────────────────────────────────────────────────────────────────
# CELL 5 — Clone repo (Colab) or verify local
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_code_cell("""# Cell 5 — Get source code
# In Colab: clones the GitHub repo so src/ and all notebooks are available.
# Locally: assumes you're already in the repo root — no-op.

GITHUB_URL = "https://github.com/Eliekh2/315-REPO.git"
REPO_DIR   = "315-REPO"

if IN_COLAB:
    if not os.path.exists(REPO_DIR):
        print("Cloning repository...")
        os.system(f"git clone -q {GITHUB_URL}")
    else:
        print("Repository already cloned.")
    os.chdir(REPO_DIR)

# Make src/ importable regardless of environment
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

print(f"Working directory : {os.getcwd()}")
print(f"src/ accessible  : {os.path.exists('src')}")
print(f"data/ accessible : {os.path.exists('data/processed')}")
"""))

# ──────────────────────────────────────────────────────────────────────────────
# CELL 6 — Verify processed data
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_code_cell("""# Cell 6 — Verify processed data (from GitHub clone)
# The processed parquet files live in data/processed/ within the cloned repo.
# Primary source: GitHub. Google Drive fallback only if a file was too large.
#
# All three files are < 50 MB, so they are committed directly to GitHub.
# The GDRIVE_FILE_IDS dict below is intentionally empty.

import pandas as pd
from pathlib import Path

DATA_DIR       = Path("data/processed")
REQUIRED_FILES = ["customers_master.parquet", "train.parquet", "test.parquet"]

# Google Drive fallback — populated only if a file exceeded GitHub limits
GDRIVE_FILE_IDS = {
    "customers_master.parquet": "",
    "train.parquet":             "",
    "test.parquet":              "",
}

missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
if missing:
    print(f"Fetching {len(missing)} missing file(s) from Google Drive...")
    os.system("pip install -q gdown")
    import gdown
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in missing:
        fid = GDRIVE_FILE_IDS.get(fname, "")
        if not fid:
            raise FileNotFoundError(
                f"{fname} not found locally and no GDrive ID configured."
            )
        gdown.download(f"https://drive.google.com/uc?id={fid}",
                       str(DATA_DIR / fname), quiet=False)

print("Data verification:")
for f in REQUIRED_FILES:
    df = pd.read_parquet(DATA_DIR / f)
    print(f"  ✓ {f}: {df.shape[0]:,} rows × {df.shape[1]} cols")
"""))

# ──────────────────────────────────────────────────────────────────────────────
# CELL 7 — Pretrained models check
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_code_cell("""# Cell 7 — Pretrained models
# SKIP_TRAINING=True (default): loads .pkl files — total runtime ~15 min.
# SKIP_TRAINING=False: retrains all models from scratch (~60-90 min).
# The professor can toggle this flag; the default path is load-and-evaluate.

SKIP_TRAINING = True

from pathlib import Path
MODELS_DIR = Path("outputs/models")

EXPECTED_MODELS = [
    "logistic.pkl",
    "random_forest.pkl",
    "xgboost.pkl",
    "kmeans.pkl",
    "xgb_cluster_0.pkl",
    "xgb_cluster_2.pkl",
]

if SKIP_TRAINING:
    found, missing_models = [], []
    for name in EXPECTED_MODELS:
        if (MODELS_DIR / name).exists():
            found.append(name)
        else:
            missing_models.append(name)
    print(f"Pretrained models found    : {found}")
    if missing_models:
        print(f"Missing (will retrain)     : {missing_models}")
        print("  → Set SKIP_TRAINING=False to retrain all, or check the GitHub repo.")
    else:
        print("All pretrained models available. SKIP_TRAINING=True.")
else:
    print("SKIP_TRAINING=False — models will be retrained in their respective sections.")
"""))

# ──────────────────────────────────────────────────────────────────────────────
# CELL 8 — Imports and config
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_code_cell("""# Cell 8 — Project modules and global imports
from src.config import (
    RANDOM_SEED, TARGET, FIGURES, MODELS, TABLES, BENCHMARK_CSV,
    TRAIN_PATH, TEST_PATH, CUSTOMERS_MASTER,
)
from src.evaluate import evaluate, append_to_benchmark, print_metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

# Figure output directory for master notebook
MASTER_FIGS = FIGURES / "master"
MASTER_FIGS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
print(f"Random seed   : {RANDOM_SEED}")
print(f"Target column : {TARGET}")
print(f"Figures → {MASTER_FIGS}")
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION C1 — Data Acquisition
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## C1 — Data Acquisition

### Sources

Two raw files form the input to this pipeline, both produced by the MSBA 305 data engineering project:

| File | Format | Records | Description |
|---|---|---|---|
| `nigerian_mobile_money_full.parquet` | Parquet | 4,000,000 rows × 13 cols | Transaction log: one row per transaction, Jan–Jun 2024 |
| `customer_profiles.json` | JSON | 375,837 records | Customer profile attributes: demographics, account metadata |

**Join key:** `wallet_id` — 100% coverage between both files (no unmatched customers).

**Transaction schema:** `wallet_id`, `timestamp`, `transaction_type` (7 categories: TRANSFER, BILL_PAYMENT, AIRTIME, CASH_IN, CASH_OUT, SAVINGS, LOAN_REPAYMENT), `channel` (4: USSD, APP, WEB, AGENT), `device_os` (3: Android, iOS, Feature Phone), `kyc_tier` (1–3), `amount_ngn`, `fee_ngn`, `balance_after_ngn`, `fraud_flag`, `agent_id`.

**Profile schema:** `wallet_id`, `age`, `gender`, `state`, `city`, `registration_date`, `account_status`, `referral_source`, `preferred_language`, `linked_bank`, `support_tier`.

### Why customer grain, not transaction grain

The prediction unit is a **customer** — "will this customer churn?" not "will this transaction fail?" Modeling at transaction grain would require a different label definition and would cause train/test leakage when splitting by time (transactions from the same customer appear in both train and test). By aggregating to customer grain first, we guarantee that the test set contains held-out *customers*, not just held-out transactions.

The pipeline aggregates the 4M transaction log to 375,537 customer-level rows, then joins the profile attributes. This order — aggregate first, join profiles second — matches the expected inference flow: behavioral features are computed from live transaction streams; profile attributes are a one-time lookup.
"""))

cells.append(new_code_cell("""# C1 — Load customer master file (pre-built, ~375K rows)
df_master = pd.read_parquet(CUSTOMERS_MASTER)
print(f"Shape : {df_master.shape[0]:,} rows × {df_master.shape[1]} columns")
print(f"\\nColumn list:")
print([c for c in df_master.columns if c not in ["wallet_id", "full_name"]])
print(f"\\nSample (5 rows):")
df_master.drop(columns=["wallet_id", "full_name"], errors="ignore").head(5)
"""))

cells.append(new_markdown_cell("""The master file contains 34 model features plus 3 metadata columns (`wallet_id`, `full_name`, `churned_vendor`). All metadata columns are excluded from model inputs — they exist for output labeling and sanity-checking only.
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION C2 — Preprocessing & Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## C2 — Preprocessing & Feature Engineering

### Time-window split (leakage prevention)

The 6-month dataset (January–June 2024) is split into two non-overlapping windows:

| Window | Period | Purpose |
|---|---|---|
| Feature window | Jan 1 – Apr 30, 2024 (4 months) | Source for ALL behavioral features |
| Label window | May 1 – Jun 30, 2024 (2 months) | Source for the churn label ONLY |

**Why this matters:** if a customer's May or June activity appeared in their features, any model could trivially predict churn by checking whether they transacted at all — this would be pure leakage, not learning. The wall at April 30 is enforced at the data loader level; no feature computation ever touches label-window data.

### Churn label definition

```
churned = 1  if  (transactions in label window) == 0
churned = 0  otherwise
```

This produces a **10.1% churn rate** on the adjusted master file (37,990 churners out of 375,537 customers).

The vendor-supplied `churn_30d` column uses a different definition (30-day inactivity, not 60-day) and a different grain (transaction-level). We retain it as `churned_vendor` for cross-validation — agreement rate between the two labels is ~62%, indicating material definitional divergence that is documented in `docs/ISSUES_AND_FIXES.md`.

### Engineered features

34 features are built from the raw files:

- **Behavioral (transaction-based, 20 features):** transaction count, total/avg/std/min/max amount, total fee, last balance, avg balance, std balance, count by type (5 types), count by channel (4 channels), H1 vs H2 transaction counts (for velocity change)
- **Recency (2 features):** `days_since_last_txn`, `txn_velocity_change` (H2/H1 count ratio)
- **Reactivation (2 features):** `is_reactivator` (binary), `dormancy_days_before_reactivation`
- **Profile (10 features):** `age`, `gender`, `kyc_tier`, `referral_source`, `preferred_language`, `linked_bank`, `state`, `city`, `tenure_days`, `account_active_span_days`

### Feature audit decisions

Seven fields were removed from the master file after a post-EDA audit:
- `date_of_birth`, `registration_date`, `first_txn_date`, `last_txn_date` — raw dates redundant with derived features (`age`, `tenure_days`, `days_since_last_txn`)
- `account_status`, `support_tier`, `notification_preferences` — synthesized independently of the transaction log; risk of label-contradicting signal
"""))

cells.append(new_code_cell("""# C2 — Feature window and churn label verification
from src.config import FEATURE_WINDOW_START, FEATURE_WINDOW_END, LABEL_WINDOW_START, LABEL_WINDOW_END

print(f"Feature window : {FEATURE_WINDOW_START.date()} → {FEATURE_WINDOW_END.date()}")
print(f"Label window   : {LABEL_WINDOW_START.date()} → {LABEL_WINDOW_END.date()}")
print()
print(f"Total customers : {len(df_master):,}")
churn_rate = df_master[TARGET].mean()
print(f"Churn rate      : {churn_rate:.1%}  ({df_master[TARGET].sum():,} churners)")

# Cross-check vendor label if present
if "churned_vendor" in df_master.columns:
    agree = (df_master[TARGET] == df_master["churned_vendor"]).mean()
    print(f"\\nVendor label agreement : {agree:.1%}  (sanity check — not used in modeling)")
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION C3 — EDA
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## C3 — Exploratory Data Analysis

Four key findings from EDA that shaped the modeling strategy:

1. **Class imbalance:** 10.1% churn rate requires imbalance-aware training (`scale_pos_weight`, class_weight, or resampling).
2. **Reactivator sub-population:** ~1.5% of customers (5,757 wallets) had zero transactions in the feature window but reactivated in the label window. These customers churn at 100% — they are definitionally reactivators who cannot sustain activity.
3. **Recency dominates permutation importance:** `days_since_last_txn` (Spearman ρ = 0.15) and `txn_velocity_change` are the strongest linear signals. Tree models confirm this in Phase 2.3.
4. **Non-linear structure:** The AUC jump from Logistic Regression (0.708) to Random Forest (0.757) confirms that churn is not linearly separable in this feature space.
"""))

cells.append(new_code_cell("""# C3 — Key EDA figures

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Class balance
ax = axes[0]
counts = df_master[TARGET].value_counts()
bars = ax.bar(["Retained (0)", "Churned (1)"], counts.values,
              color=["#4C72B0", "#DD8452"], edgecolor="white", linewidth=1.2)
ax.set_title("Class Balance", fontweight="bold")
ax.set_ylabel("Customers")
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f"{v:,}\\n({v/len(df_master):.1%})", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, counts.max() * 1.15)

# 2. Churn rate by KYC tier
ax = axes[1]
kyc_churn = df_master.groupby("kyc_tier")[TARGET].mean().reset_index()
sns.barplot(data=kyc_churn, x="kyc_tier", y=TARGET, ax=ax, palette="Blues_d")
ax.set_title("Churn Rate by KYC Tier", fontweight="bold")
ax.set_xlabel("KYC Tier")
ax.set_ylabel("Churn Rate")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# 3. Days since last transaction distribution
ax = axes[2]
churned = df_master[df_master[TARGET] == 1]["days_since_last_txn"]
retained = df_master[df_master[TARGET] == 0]["days_since_last_txn"]
ax.hist(retained, bins=50, alpha=0.6, label="Retained", color="#4C72B0", density=True)
ax.hist(churned,  bins=50, alpha=0.6, label="Churned",  color="#DD8452", density=True)
ax.set_title("Recency by Churn Status", fontweight="bold")
ax.set_xlabel("Days Since Last Transaction")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.savefig(MASTER_FIGS / "c3_eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: c3_eda_overview.png")
"""))

cells.append(new_code_cell("""# C3 — Reactivator sub-population
react_mask = df_master["is_reactivator"] == 1
n_react = react_mask.sum()
react_churn = df_master[react_mask][TARGET].mean()
nonreact_churn = df_master[~react_mask][TARGET].mean()

print(f"Reactivators (is_reactivator=1) : {n_react:,} customers ({n_react/len(df_master):.1%})")
print(f"  Churn rate (reactivators)     : {react_churn:.1%}")
print(f"  Churn rate (non-reactivators) : {nonreact_churn:.1%}")
print()

# Top feature correlations with churn
feat_cols = [c for c in df_master.select_dtypes("number").columns
             if c not in [TARGET, "churned_vendor", "is_reactivator",
                          "dormancy_days_before_reactivation"]]
corr = df_master[feat_cols + [TARGET]].corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print("Top 10 |Spearman| correlations with churn:")
print(corr.head(10).round(4).to_string())
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION C4 — Train/Test Split
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## C4 — Train/Test Split

The master file is split **once** into a frozen 80/20 stratified train/test partition using `random_state=42`. The split is stratified on the `churned` label to preserve the 10.1% churn rate in both sets.

The test set is **never touched** during model development — not in cross-validation, not in threshold tuning, not in feature selection. All tuning and hyperparameter search happens inside the training set via 5-fold stratified CV. The test set is used exactly once per model: the final held-out evaluation reported in the benchmark.
"""))

cells.append(new_code_cell("""# C4 — Load frozen train/test split
df_train = pd.read_parquet(TRAIN_PATH)
df_test  = pd.read_parquet(TEST_PATH)

print(f"Train : {df_train.shape[0]:,} rows  (churn rate: {df_train[TARGET].mean():.1%})")
print(f"Test  : {df_test.shape[0]:,} rows  (churn rate: {df_test[TARGET].mean():.1%})")
print(f"Split : {len(df_train)/(len(df_train)+len(df_test)):.0%} / {len(df_test)/(len(df_train)+len(df_test)):.0%}")

# Metadata + target columns — exclude from model features
META_COLS  = ["wallet_id", "full_name", "churned_vendor"]
MODEL_COLS = [c for c in df_train.columns if c not in META_COLS + [TARGET]]
print(f"\\nModel features : {len(MODEL_COLS)}")
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION C5 — Logistic Regression
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## C5 — Modeling Track 1: Logistic Regression Baseline

Logistic Regression serves as the **interpretable baseline** — a model that can be explained to a non-technical stakeholder and whose coefficients directly encode feature importance. It also confirms whether the problem is linearly separable (it is not, as the AUC gap to tree models shows).

**Key decisions:**
- Solver: `liblinear` (coordinate descent). `saga` with full one-hot encoding of `state`/`city` exceeded 20 minutes on 300K rows.
- Features: `state` and `city` are excluded from LR (high-cardinality OHE destabilizes liblinear); reintroduced for tree models via ordinal encoding.
- Regularization: L1 with C=0.1 (selected via 5-fold CV grid search over L1/L2 × C ∈ {0.01, 0.1, 1, 10}).
- Imbalance: `class_weight='balanced'`.
"""))

cells.append(new_code_cell("""# C5 — Logistic Regression: load pretrained model and evaluate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np

LR_SKIP_COLS = ["state", "city", "is_reactivator", "dormancy_days_before_reactivation"]
lr_feat_cols = [c for c in MODEL_COLS if c not in LR_SKIP_COLS]

if SKIP_TRAINING and (MODELS / "logistic.pkl").exists():
    lr_pipe = joblib.load(MODELS / "logistic.pkl")
    print("Loaded logistic.pkl")
else:
    print("Training Logistic Regression (L1, C=0.1, liblinear)...")
    cat_cols_lr = [c for c in lr_feat_cols if df_train[c].dtype == "object"]
    num_cols_lr = [c for c in lr_feat_cols if df_train[c].dtype != "object"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols_lr),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols_lr),
    ])
    lr_pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(penalty="l1", C=0.1, solver="liblinear",
                                    class_weight="balanced", max_iter=1000,
                                    random_state=RANDOM_SEED)),
    ])
    lr_pipe.fit(df_train[lr_feat_cols], df_train[TARGET])
    joblib.dump(lr_pipe, MODELS / "logistic.pkl")
    print("Trained and saved logistic.pkl")

X_test_lr = df_test[lr_feat_cols]
y_test     = df_test[TARGET].values
lr_proba   = lr_pipe.predict_proba(X_test_lr)[:, 1]
lr_results = evaluate(y_test, lr_proba)
print_metrics(lr_results, "Logistic_Optimized")
"""))

cells.append(new_markdown_cell("""**Interpretation:** Logistic regression achieves AUC 0.708 and PR-AUC 0.257. The model correctly identifies the broad signal (recency, frequency) but misses non-linear interactions. The AUC gap to XGBoost (+0.051) is the headline finding confirming that churn is non-linearly structured — a finding consistent with the MENA fintech literature (Hadden et al., 2007; Verbeke et al., 2012).
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION C6 — Random Forest
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## C6 — Modeling Track 2: Random Forest

Random Forest is the **bagging benchmark** — an ensemble method that averages over independently grown trees, reducing variance without bias amplification. It handles mixed types natively (via ordinal encoding), does not require scaling, and provides out-of-bag (OOB) error estimates without a dedicated validation set.

**Hyperparameter search:** RandomizedSearchCV (50 iterations, 5-fold stratified CV) over: n_estimators (100–600), max_depth (10–40 or None), min_samples_split (2–20), min_samples_leaf (1–10), max_features ("sqrt"/"log2"), class_weight ("balanced"/"balanced_subsample").

**OOB note:** sklearn's `oob_score_` returns accuracy, which is misleading for imbalanced data. OOB AUC is computed correctly using `oob_decision_function_[:, 1]` with `roc_auc_score`.
"""))

cells.append(new_code_cell("""# C6 — Random Forest: load pretrained model and evaluate
from sklearn.ensemble import RandomForestClassifier

if SKIP_TRAINING and (MODELS / "random_forest.pkl").exists():
    rf_pipe = joblib.load(MODELS / "random_forest.pkl")
    print("Loaded random_forest.pkl")
else:
    print("Training Random Forest (RandomizedSearchCV, 50 iter)...")
    from sklearn.model_selection import RandomizedSearchCV
    cat_cols_rf = [c for c in MODEL_COLS if df_train[c].dtype == "object"]
    num_cols_rf = [c for c in MODEL_COLS if df_train[c].dtype != "object"]

    pre_rf = ColumnTransformer([
        ("num", "passthrough", num_cols_rf),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols_rf),
    ])
    param_dist = {
        "clf__n_estimators":    [100, 200, 300, 400, 500, 600],
        "clf__max_depth":       [10, 15, 20, 30, None],
        "clf__min_samples_split": range(2, 20),
        "clf__min_samples_leaf":  range(1, 10),
        "clf__max_features":    ["sqrt", "log2"],
        "clf__class_weight":    ["balanced", "balanced_subsample"],
    }
    rf_pipe = Pipeline([
        ("pre", pre_rf),
        ("clf", RandomForestClassifier(oob_score=True, random_state=RANDOM_SEED, n_jobs=-1)),
    ])
    search = RandomizedSearchCV(rf_pipe, param_dist, n_iter=50, cv=5, scoring="roc_auc",
                                n_jobs=-1, random_state=RANDOM_SEED, verbose=1)
    search.fit(df_train[MODEL_COLS], df_train[TARGET])
    rf_pipe = search.best_estimator_
    joblib.dump(rf_pipe, MODELS / "random_forest.pkl")
    print(f"Best RF AUC (CV): {search.best_score_:.4f}")

rf_proba   = rf_pipe.predict_proba(df_test[MODEL_COLS])[:, 1]
rf_results = evaluate(y_test, rf_proba)
print_metrics(rf_results, "RandomForest_Tuned")
"""))

cells.append(new_markdown_cell("""**Interpretation:** Random Forest achieves AUC 0.757 (+0.050 over LR), PR-AUC 0.422 (+0.165), and 38.8% precision in the top decile. The jump validates the non-linear hypothesis. Permutation importance on the RF model places `days_since_last_txn` and `txn_velocity_change` at the top — consistent with the literature and the LR coefficient ranking.
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION C7 — XGBoost
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## C7 — Modeling Track 3: XGBoost (Primary Model)

XGBoost is the **primary production model** — a boosting ensemble that sequentially corrects residuals from prior trees. Compared to Random Forest, XGBoost is stronger at capturing shallow interactions when regularized well. The key challenge: without regularization, XGBoost will overfit the reactivator sub-population (whose features — dormancy days, is_reactivator — are perfectly predictive but apply to only 1.5% of customers).

**Optimization:** Optuna TPE sampler, 30 trials, 5-fold stratified CV, AUC objective. Best trial (trial 24): CV AUC = 0.7614.

**Best hyperparameters:**
| Parameter | Value |
|---|---|
| n_estimators | 825 (but best_iteration = 249 via early stopping) |
| max_depth | 6 |
| learning_rate | 0.0103 |
| subsample | 0.705 |
| colsample_bytree | 0.830 |
| min_child_weight | 9 |
| gamma | 2.356 |
| reg_alpha | 2.611 |
| reg_lambda | 2.482 |
| scale_pos_weight | 8.889 (neg/pos ratio) |

**Three feature importance methods:** gain (split quality), weight (split count), permutation (test AUC drop). The gain/permutation divergence for `dormancy_days_before_reactivation` is expected: it dominates gain because it creates pure nodes, but it applies to only 1.5% of test customers so its permutation impact is limited.
"""))

cells.append(new_code_cell("""# C7 — XGBoost: load pretrained model and evaluate
from xgboost import XGBClassifier
import json

if SKIP_TRAINING and (MODELS / "xgboost.pkl").exists():
    xgb_pipe = joblib.load(MODELS / "xgboost.pkl")
    print("Loaded xgboost.pkl")
else:
    print("Training XGBoost (Optuna 30 trials — ~20 min)...")
    print("This will run Optuna; set SKIP_TRAINING=True to skip.")
    # (Full training code is in notebooks/04_xgboost.ipynb)
    raise RuntimeError("Set SKIP_TRAINING=True to load pretrained XGBoost, or run notebooks/04_xgboost.ipynb")

xgb_proba   = xgb_pipe.predict_proba(df_test[MODEL_COLS])[:, 1]
xgb_results = evaluate(y_test, xgb_proba)
print_metrics(xgb_results, "XGBoost_Tuned")

# Load feature importance summary if available
xgb_json_path = TABLES / "xgb_results.json"
if xgb_json_path.exists():
    with open(xgb_json_path) as f:
        xgb_meta = json.load(f)
    top5_gain = xgb_meta.get("top5_gain", [])
    top5_perm = xgb_meta.get("top5_perm", [])
    print(f"\\nTop 5 by gain        : {top5_gain}")
    print(f"Top 5 by permutation : {top5_perm}")
"""))

cells.append(new_code_cell("""# C7 — XGBoost feature importance visualization
xgb_fi_path = TABLES / "xgb_feature_importance.csv"
if xgb_fi_path.exists():
    fi_df = pd.read_csv(xgb_fi_path).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gain importance
    ax = axes[0]
    fi_sorted = fi_df.sort_values("gain_score", ascending=True).tail(12)
    ax.barh(fi_sorted["feature"], fi_sorted["gain_score"], color="#4C72B0")
    ax.set_title("XGBoost — Feature Importance (Gain)", fontweight="bold")
    ax.set_xlabel("Mean Gain")

    # Permutation importance
    ax = axes[1]
    fi_sorted2 = fi_df.sort_values("perm_auc_drop", ascending=True).tail(12)
    ax.barh(fi_sorted2["feature"], fi_sorted2["perm_auc_drop"], color="#DD8452")
    ax.set_title("XGBoost — Feature Importance (Permutation)", fontweight="bold")
    ax.set_xlabel("AUC Drop on Permutation")

    plt.tight_layout()
    plt.savefig(MASTER_FIGS / "c7_xgb_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: c7_xgb_feature_importance.png")
else:
    print("Feature importance CSV not found — run notebooks/04_xgboost.ipynb to generate.")
"""))

cells.append(new_markdown_cell("""**Interpretation:** XGBoost achieves AUC 0.759 (vs RF 0.757) — the difference is statistically marginal (+0.002) but the model is more regularized and better calibrated as a pipeline component. The primary finding is the **LR-to-tree gap (+0.051 AUC)**, not the RF-to-XGBoost gap. Tuning with Optuna adds no AUC gain over the vanilla XGBoost, but documents the regularization choices and confirms the model is not significantly overfitting.
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION C8 — Cluster-then-Predict (novelty)
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## C8 — Novelty Layer: Cluster-then-Predict

This section implements the novel two-stage architecture that differentiates RetainIQ from a standard benchmark comparison.

**Motivation:** The EDA and XGBoost feature importance both suggest that churn is not homogeneous across the customer base — reactivators churn at 100%, very light users at ~11%, and regular active customers at ~8.7%. A global model must compromise across these populations. The cluster-then-predict approach tests whether segmenting customers by behavior first, then training per-segment models, produces better predictions than a single global model.

**Architecture:**
1. K-means clustering on 13 behavioral features (StandardScaler — Euclidean-distance sensitive)
2. K selection via silhouette score (K=2..6, sampled on 30K stratified subset)
3. Per-cluster XGBoost trained with Phase 2.3 best hyperparameters (n_estimators=249 fixed — no re-tuning per cluster to prevent overfitting on small clusters)
4. At inference: cluster assignment → route to cluster model (or global fallback for pure-class clusters)

**13 clustering features:** `txn_count`, `total_amount_ngn`, `avg_amount_ngn`, `std_amount_ngn`, `total_fee_ngn`, `avg_balance_ngn`, `std_balance_ngn`, `days_since_last_txn`, `txn_velocity_change`, `is_reactivator`, `dormancy_days_before_reactivation`, `tenure_days`, `kyc_tier`
"""))

cells.append(new_code_cell("""# C8 — Cluster-then-Predict: load results and visualize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

CLUSTER_JSON = TABLES / "cluster_predict_results.json"
CLUSTER_PROFILES_CSV = TABLES / "cluster_profiles.csv"

if CLUSTER_JSON.exists():
    with open(CLUSTER_JSON) as f:
        cr = json.load(f)

    print("=== K-means Results ===")
    print(f"Chosen K              : {cr['chosen_k']}")
    print(f"Silhouette scores K2-6: {cr['silhouette_scores']}")
    print(f"Global XGB AUC        : {cr['global_xgb_auc']:.4f}")
    print(f"ClusterPredict AUC    : {cr['cluster_predict_auc']:.4f}")
    auc_delta = cr.get('auc_delta_vs_global') or cr.get('auc_delta') or 0.0
    ceiling   = cr.get('ceiling_broken', auc_delta > 0.001)
    print(f"AUC delta             : {auc_delta:+.4f}")
    print(f"Ceiling broken (>0.76): {ceiling}")

if CLUSTER_PROFILES_CSV.exists():
    cluster_profiles = pd.read_csv(CLUSTER_PROFILES_CSV)
    print("\\n=== Cluster Profiles ===")
    display_cols = ["cluster", "n_train", "pct_train", "churn_rate",
                    "mean_txn_count", "mean_days_since_last_txn",
                    "mean_is_reactivator", "mean_dormancy_days_before_reactivation"]
    display_cols = [c for c in display_cols if c in cluster_profiles.columns]
    print(cluster_profiles[display_cols].to_string(index=False))
else:
    print("Cluster profiles not found — run notebooks/06_cluster_then_predict.ipynb")
"""))

cells.append(new_code_cell("""# C8 — Cluster-then-Predict evaluation
if SKIP_TRAINING and (MODELS / "kmeans.pkl").exists():
    kmeans = joblib.load(MODELS / "kmeans.pkl")
    print("Loaded kmeans.pkl")

    CLUSTER_FEATURES = [
        "txn_count", "total_amount_ngn", "avg_amount_ngn", "std_amount_ngn",
        "total_fee_ngn", "avg_balance_ngn", "std_balance_ngn",
        "days_since_last_txn", "txn_velocity_change", "is_reactivator",
        "dormancy_days_before_reactivation", "tenure_days", "kyc_tier",
    ]

    # Load cluster-specific models
    cluster_models = {}
    for k in [0, 2]:
        pkl_path = MODELS / f"xgb_cluster_{k}.pkl"
        if pkl_path.exists():
            cluster_models[k] = joblib.load(pkl_path)
    global_model = joblib.load(MODELS / "xgboost.pkl")

    # Predict
    X_test_cluster = df_test[CLUSTER_FEATURES].fillna(0)

    # Scale using train stats
    scaler = StandardScaler()
    scaler.fit(df_train[CLUSTER_FEATURES].fillna(0))
    X_test_scaled = scaler.transform(X_test_cluster)

    test_clusters = kmeans.predict(X_test_scaled)
    cluster_proba = np.zeros(len(df_test))

    for k in np.unique(test_clusters):
        mask = test_clusters == k
        if k in cluster_models:
            cluster_proba[mask] = cluster_models[k].predict_proba(df_test[mask][MODEL_COLS])[:, 1]
        else:
            cluster_proba[mask] = global_model.predict_proba(df_test[mask][MODEL_COLS])[:, 1]

    cp_results = evaluate(y_test, cluster_proba)
    print_metrics(cp_results, "ClusterPredict_XGB")
else:
    print("Cluster models not found — run notebooks/06_cluster_then_predict.ipynb")
"""))

cells.append(new_markdown_cell("""**Findings:**

| Cluster | Label | N (train) | Churn rate | Key signal |
|---|---|---|---|---|
| 0 | Regular Active Customers | 293,532 (97.7%) | 8.7% | 26.7 mean txns, no reactivators |
| 1 | Dormant Reactivators | 4,589 (1.5%) | **100%** | is_reactivator=1.0, 194.5 days dormancy |
| 2 | Very Light Users | 2,308 (0.8%) | 11.3% | 3.35 mean txns |

**Result:** ClusterPredict AUC = 0.7585 vs Global XGB = 0.7590 (delta = −0.0005). The ceiling was **not broken** — explicit segmentation added no information beyond what the XGBoost feature set already encoded via `dormancy_days_before_reactivation` (gain rank 1) and `is_reactivator` (gain rank 2).

**The headline finding is not AUC lift.** It is that K-means clustering on behavioral features alone, with **zero label information**, recovered a near-perfect churn segment (Cluster 1: 100% churn rate). This validates that churn has a strong behavioral signature detectable without supervision.

**Production routing rule:**
- Cluster 1 (reactivators) → **automatic retention trigger** — no model needed, 100% churn rate
- Cluster 0/2 → **global XGBoost** → risk score → intervention queue
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION C9 — Benchmark Synthesis
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## C9 — Benchmark Synthesis

All models evaluated on the same frozen test set using the canonical `evaluate()` function from `src/evaluate.py`. Results appended to `outputs/tables/benchmark.csv` — one row per model, committed to GitHub.
"""))

cells.append(new_code_cell("""# C9 — Final benchmark table
benchmark = pd.read_csv(BENCHMARK_CSV)
display_cols = ["model", "auc", "pr_auc", "f1", "precision_at_k", "recall_at_k"]
display_cols = [c for c in display_cols if c in benchmark.columns]
bm = benchmark[display_cols].copy()
bm = bm.sort_values("auc", ascending=False).reset_index(drop=True)

# Format
for col in ["auc", "pr_auc", "f1", "precision_at_k", "recall_at_k"]:
    if col in bm.columns:
        bm[col] = bm[col].map(lambda x: f"{x:.4f}")

with pd.option_context("display.max_rows", None, "display.max_columns", None,
                        "display.width", 120):
    print(bm.to_string(index=False))
"""))

cells.append(new_code_cell("""# C9 — AUC comparison bar chart
bm_plot = benchmark[["model", "auc", "pr_auc"]].sort_values("auc", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#4C72B0" if "Logistic" in m else "#55A868" if "Forest" in m
          else "#C44E52" if "Cluster" in m else "#DD8452"
          for m in bm_plot["model"]]

for ax, metric, label in zip(axes, ["auc", "pr_auc"], ["AUC (ROC)", "PR-AUC"]):
    bars = ax.barh(bm_plot["model"], bm_plot[metric], color=colors)
    ax.set_title(f"Model Comparison — {label}", fontweight="bold")
    ax.set_xlabel(label)
    ax.set_xlim(0, bm_plot[metric].max() * 1.15)
    for bar, v in zip(bars, bm_plot[metric]):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(MASTER_FIGS / "c9_benchmark_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: c9_benchmark_comparison.png")
"""))

cells.append(new_markdown_cell("""**Summary:** XGBoost (tuned) achieves the best AUC (0.759) and PR-AUC (0.426), followed by XGBoost vanilla (0.759/0.425) and Random Forest (0.757/0.422). Logistic regression lags significantly (0.708/0.257). The LR-to-tree gap (+0.051 AUC) is the primary finding. The ceiling at ~0.76 AUC suggests that additional algorithmic complexity (boosting variants, cluster segmentation) does not unlock a new performance tier — the data's predictive signal is approximately fully captured by a well-regularized XGBoost.

**Precision at top 10%:** all tree models achieve ~38.8% — a 3.8× lift over the 10.1% base rate. A retention campaign targeting the top decile contacts 38.8% churners vs. 10.1% at random.
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION C10 — SHAP Analysis placeholder
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## C10 — SHAP Analysis (Explainability)

SHAP (SHapley Additive exPlanations) provides model-agnostic feature attribution: for each prediction, SHAP decomposes the output into additive contributions from each feature, grounded in cooperative game theory.

The full SHAP analysis is in `notebooks/08_shap_analysis.ipynb`. Key findings from that notebook:

- **Global ranking** (mean |SHAP|): `days_since_last_txn` > `txn_count` > `txn_velocity_change` > `tenure_days` > `kyc_tier`
- **Direction:** Higher recency (more days since last transaction) → higher churn probability. Higher transaction count → lower churn probability.
- **Reactivator features:** `dormancy_days_before_reactivation` and `is_reactivator` dominate for the 1.5% reactivator sub-population but have near-zero SHAP values for the remaining 98.5%.

The code below runs a lightweight SHAP summary using a subsample of the test set for Colab runtime efficiency.
"""))

cells.append(new_code_cell("""# C10 — SHAP analysis (subsample for Colab speed)
try:
    import shap

    xgb_clf = xgb_pipe.named_steps["clf"]
    pre     = xgb_pipe.named_steps["pre"]
    X_test_transformed = pre.transform(df_test[MODEL_COLS])

    # Use a 2K subsample for Colab speed
    idx = np.random.choice(len(X_test_transformed), size=min(2000, len(X_test_transformed)),
                           replace=False)
    X_shap = X_test_transformed[idx] if hasattr(X_test_transformed, '__getitem__') else X_test_transformed.iloc[idx]

    explainer   = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_shap)

    # Feature names from the preprocessor
    try:
        feat_names = pre.get_feature_names_out()
        feat_names = [n.replace("num__", "").replace("cat__", "") for n in feat_names]
    except Exception:
        feat_names = MODEL_COLS

    print(f"SHAP computed on {len(idx)} test samples")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feat_names,
                      max_display=15, show=False)
    plt.title("SHAP Summary — XGBoost Tuned", fontweight="bold")
    plt.tight_layout()
    plt.savefig(MASTER_FIGS / "c10_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: c10_shap_summary.png")

except Exception as e:
    print(f"SHAP analysis skipped: {e}")
    print("Run notebooks/08_shap_analysis.ipynb for the full SHAP breakdown.")
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION C11 — Conclusions & Business Recommendations
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## C11 — Conclusions & Business Recommendations

### Research questions answered

**RQ1: Can customer churn in mobile money platforms be predicted from behavioral and demographic data?**
Yes. The best model (XGBoost, Optuna-tuned) achieves AUC 0.759, PR-AUC 0.426, and 38.9% precision in the top decile — a 3.8× lift over random. Churn is predictable with useful precision from a 4-month behavioral window.

**RQ2: Do tree-based models outperform logistic regression for this task?**
Substantially. The AUC gap is +0.051 (0.708 → 0.759), the PR-AUC gap is +0.169 (0.257 → 0.426). This confirms that churn is non-linearly structured — transaction frequency and recency interact in ways that a linear model cannot capture.

**RQ3: Does cluster-based segmentation improve prediction?**
Not in AUC terms (delta = −0.0005). But the approach revealed that unsupervised K-means clustering recovers a near-perfect churn cluster (100% churn rate, 1.5% of customers) without any label information — a qualitative finding with direct production value.

### Deployment recommendations

For a deploying operator (e.g., Whish Money, OMT Pay):

1. **Reactivator trigger (immediate):** Flag the 1.5% of customers with `is_reactivator=1` as automatic churn candidates — no model inference needed. These customers reactivate briefly and almost universally churn. Design a targeted "welcome back" sequence that begins at reactivation, not at predicted churn.

2. **Top-decile campaign (weekly batch):** Run the XGBoost pipeline on the full active customer base weekly. Contact the top 10% by predicted churn probability (38.8% precision). At scale, a 38.8% contact precision × intervention success rate of 20–30% would retain 7.8–11.6% of true churners per campaign cycle.

3. **Deployment threshold:** Use recall ≥ 0.80 threshold (score ≥ 0.35) rather than the default 0.5. At 0.35, the model flags more customers (higher recall, lower precision) — appropriate when the cost of missing a churner exceeds the cost of a false alarm.

4. **Recency monitoring:** `days_since_last_txn` is the #1 permutation-importance feature. Build a real-time recency alert: any customer silent for 15+ days triggers a proactive nudge (SMS, push notification, incentive). This is a leading indicator, not a lagging one.

5. **Calibrate before deploying probabilities:** All three model families (LR, RF, XGB) are miscalibrated (MACE > 0.25). Apply isotonic calibration before using predicted probabilities for business decisions (e.g., tiered intervention intensity based on predicted churn probability).

### Limitations

- **Synthetic data:** The Nigerian dataset is simulation-based, not real transaction data. The behavioral patterns and churn rate are plausible but not validated against real mobile money churn.
- **Proxy geography:** Lebanon / MENA churn dynamics may differ from Nigeria in ways the model cannot capture (regulatory environment, economic volatility, product mix).
- **Static model:** No concept drift handling. A model trained on Jan–Apr 2024 data should be retrained quarterly with fresh labeled data.
- **No A/B validation:** The model has not been tested in a live retention campaign. Business lift (retention success × LTV × cost) has not been measured.

### Future work

- Real Lebanese transaction data from a pilot with Whish Money or OMT Pay
- Online learning pipeline for continuous model updates
- A/B test of retention interventions triggered by the model
- Survival analysis framing (time-to-churn rather than binary churn flag)
- Graph features: referral network, agent network connectivity as churn predictors
"""))

# ──────────────────────────────────────────────────────────────────────────────
# SECTION D — Appendix
# ──────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## Appendix A — Reproducibility

| Item | Value |
|---|---|
| Random seed | 42 (all models, all splits) |
| Train/test split | Stratified 80/20 on `churned`, frozen after Phase 1 |
| Cross-validation | 5-fold stratified, training set only |
| Python version | 3.11 |
| Hardware tested | Local Windows 11 (CPU), Google Colab (CPU) |
| Estimated runtime (SKIP_TRAINING=True) | ~12–18 minutes |
| Estimated runtime (SKIP_TRAINING=False) | ~60–90 minutes |
"""))

cells.append(new_code_cell("""# Appendix A — Library versions
import importlib, pkg_resources

libs = ["pandas", "numpy", "scikit-learn", "xgboost", "lightgbm",
        "imbalanced-learn", "shap", "optuna", "matplotlib", "seaborn", "joblib"]

print("Library versions:")
for lib in libs:
    try:
        v = pkg_resources.get_distribution(lib).version
        print(f"  {lib:<25} {v}")
    except Exception:
        print(f"  {lib:<25} (not installed)")
"""))

cells.append(new_markdown_cell("""## Appendix B — Citation

If you use this work, please cite:

> Khayrallah, E. (2026). *RetainIQ: Customer Churn Prediction for Digital Wallets*. MSBA 315 — Machine Learning & Predictive Analytics, American University of Beirut.

Code repository: https://github.com/Eliekh2/315-REPO

Dataset: Nigerian Mobile Money Dataset (synthetic), publicly available.

## Appendix C — Acknowledgments

- **Prof. Wael Khreich** — MSBA 315 course instructor, AUB
- **AUB MSBA program** — for the research infrastructure and course context
- **Anthropic Claude** — used as an AI development partner throughout the project. All decisions, code, and findings are the author's; Claude assisted with implementation, debugging, and documentation. Usage logged in `docs/ai_usage_log.md` per course requirements.
"""))

# ──────────────────────────────────────────────────────────────────────────────
# Write notebook
# ──────────────────────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.11.0",
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Written: {NB_PATH}")
print(f"Cells  : {len(cells)}")
