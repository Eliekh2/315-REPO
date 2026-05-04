# -*- coding: utf-8 -*-
"""
run_xgb.py — Phase 2.3 XGBoost Training Script
================================================
Standalone script (no Jupyter) to avoid Windows fork overhead.
Runs vanilla XGB + Optuna tuning, computes feature importances,
generates all figures, saves model + results.

Run from repo root:
    .venv\Scripts\python run_xgb.py
"""
import os
import sys
import json
import time
import warnings
import logging

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

# Suppress optuna verbose logging
logging.getLogger("optuna").setLevel(logging.WARNING)

# Repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc as sk_auc,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.calibration import calibration_curve as sk_cal_curve

import xgboost as xgb
import optuna

from src.config import (
    TRAIN_PATH, TEST_PATH, FIGURES, MODELS, TABLES,
    BENCHMARK_CSV, RANDOM_SEED
)
from src.evaluate import evaluate, print_metrics

# ──────────────────────────────────────────────
# 0. Output directory
# ──────────────────────────────────────────────
FIG_DIR = FIGURES / "04_xgboost"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"[setup] Figures -> {FIG_DIR}")

# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
print("\n[1] Loading train/test...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

TARGET   = "churned"
METADATA = ["wallet_id", "full_name", "churned_vendor"]

# Exclusion list: same as RF + feature audit droplist
# date/raw columns dropped from master already; these are the remaining exclusions
FEATURE_COLS = [c for c in train.columns if c not in [TARGET] + METADATA]

CATEGORICAL_COLS = [
    "gender", "state", "city", "referral_source",
    "preferred_language", "linked_bank", "kyc_tier",
]
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

X_train = train[FEATURE_COLS]
y_train = train[TARGET].values
X_test  = test[FEATURE_COLS]
y_test  = test[TARGET].values

print(f"  Train: {train.shape}  |  Test: {test.shape}")
print(f"  Churn rate — train: {y_train.mean():.4f}  test: {y_test.mean():.4f}")
print(f"  Total features: {len(FEATURE_COLS)}  (numeric={len(NUMERIC_COLS)}, cat={len(CATEGORICAL_COLS)})")

# ──────────────────────────────────────────────
# 2. Preprocessor — OrdinalEncoder for cats, passthrough numeric
# ──────────────────────────────────────────────
print("\n[2] Building preprocessor...")
ordinal_enc = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)
preprocessor = ColumnTransformer(
    transformers=[("cat", ordinal_enc, CATEGORICAL_COLS)],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocessor.fit(X_train)
feat_names = preprocessor.get_feature_names_out()
print(f"  Preprocessor fit OK. Output features: {len(feat_names)}")

# scale_pos_weight = neg / pos (XGBoost imbalance handling)
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
scale_pos_weight = n_neg / n_pos
print(f"  scale_pos_weight = {n_neg}/{n_pos} = {scale_pos_weight:.4f}")

# ──────────────────────────────────────────────
# 3. Vanilla XGBoost
# ──────────────────────────────────────────────
print("\n[3] Training Vanilla XGBoost...")

# Internal 90/10 split for early stopping (training set only)
X_tr_enc = preprocessor.transform(X_train)
X_te_enc = preprocessor.transform(X_test)

X_sub, X_val, y_sub, y_val = train_test_split(
    X_tr_enc, y_train,
    test_size=0.10,
    stratify=y_train,
    random_state=RANDOM_SEED
)

t0 = time.time()
vanilla_clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    early_stopping_rounds=20,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    tree_method="hist",
    verbosity=0,
)
vanilla_clf.fit(
    X_sub, y_sub,
    eval_set=[(X_val, y_val)],
    verbose=False
)
elapsed_vanilla = time.time() - t0
best_iter_vanilla = vanilla_clf.best_iteration
print(f"  Vanilla done in {elapsed_vanilla:.1f}s | best_iteration={best_iter_vanilla}")

y_prob_vanilla = vanilla_clf.predict_proba(X_te_enc)[:, 1]
res_vanilla = evaluate(y_test, y_prob_vanilla)
print_metrics(res_vanilla, "XGBoost_Vanilla")

# ──────────────────────────────────────────────
# 4. safe_append — prevents duplicate benchmark rows on re-run
# ──────────────────────────────────────────────
def safe_append(model_name: str, results: dict, notes: str = "") -> None:
    """Append to benchmark.csv only if model_name not already present."""
    row = {"model": model_name, **results, "notes": notes}
    df_new = pd.DataFrame([row])
    if BENCHMARK_CSV.exists():
        df = pd.read_csv(BENCHMARK_CSV)
        if model_name in df["model"].values:
            print(f"[benchmark] {model_name} already in benchmark.csv — skipping.")
            return
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(BENCHMARK_CSV, index=False)
    print(f"[benchmark] appended {model_name} -> {BENCHMARK_CSV}")

notes_vanilla = (
    f"n_estimators={best_iter_vanilla}, max_depth=6, lr=0.1, "
    f"scale_pos_weight={scale_pos_weight:.2f}, early_stopping=20"
)
safe_append("XGBoost_Vanilla", res_vanilla, notes=notes_vanilla)

# ──────────────────────────────────────────────
# 5. Optuna Hyperparameter Search
# ──────────────────────────────────────────────
print("\n[4] Starting Optuna tuning (30 trials, 5-fold CV)...")
N_TRIALS = 30
CV_SPLITS = 5

cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)

def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "eval_metric": "auc",
        "n_jobs": -1,  # tree-level parallelism
        "random_state": RANDOM_SEED,
        "verbosity": 0,
    }

    fold_aucs = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_tr_enc, y_train)):
        X_fold_tr, X_fold_va = X_tr_enc[tr_idx], X_tr_enc[va_idx]
        y_fold_tr, y_fold_va = y_train[tr_idx], y_train[va_idx]

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_fold_tr, y_fold_tr, verbose=False)
        proba = clf.predict_proba(X_fold_va)[:, 1]
        fold_aucs.append(roc_auc_score(y_fold_va, proba))

    return float(np.mean(fold_aucs))


t0_optuna = time.time()
sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
# n_jobs=1: sequential trials (avoids Windows fork overhead)
study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)
elapsed_optuna = time.time() - t0_optuna

best_params = study.best_params
best_cv_auc = study.best_value
n_trials_run = len(study.trials)
print(f"\n  Optuna done in {elapsed_optuna/60:.1f} min | {n_trials_run} trials")
print(f"  Best CV AUC: {best_cv_auc:.4f}")
print(f"  Best params: {best_params}")

# ──────────────────────────────────────────────
# 6. Refit best params on full training set with early stopping
# ──────────────────────────────────────────────
print("\n[5] Refitting best params on full training set...")
# Use same 90/10 internal split for early stopping
best_params_fit = {
    **best_params,
    "scale_pos_weight": scale_pos_weight,
    "eval_metric": "auc",
    "early_stopping_rounds": 20,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "tree_method": "hist",
    "verbosity": 0,
}

t0_refit = time.time()
tuned_clf = xgb.XGBClassifier(**best_params_fit)
tuned_clf.fit(
    X_sub, y_sub,
    eval_set=[(X_val, y_val)],
    verbose=False
)
elapsed_refit = time.time() - t0_refit
best_iter_tuned = tuned_clf.best_iteration
print(f"  Refit done in {elapsed_refit:.1f}s | best_iteration={best_iter_tuned}")

y_prob_tuned = tuned_clf.predict_proba(X_te_enc)[:, 1]
res_tuned = evaluate(y_test, y_prob_tuned)
print_metrics(res_tuned, "XGBoost_Tuned")

notes_tuned = (
    f"Optuna 30 trials, 5-fold CV; best_cv_auc={best_cv_auc:.4f}; "
    f"best_iter={best_iter_tuned}; "
    f"n_est={best_params.get('n_estimators')}, "
    f"depth={best_params.get('max_depth')}, "
    f"lr={best_params.get('learning_rate'):.4f}, "
    f"sub={best_params.get('subsample'):.3f}, "
    f"col={best_params.get('colsample_bytree'):.3f}"
)
safe_append("XGBoost_Tuned", res_tuned, notes=notes_tuned)

# ──────────────────────────────────────────────
# 7. Feature Importance — gain, weight, permutation
# ──────────────────────────────────────────────
print("\n[6] Computing feature importances...")

booster = tuned_clf.get_booster()

# (a) gain
gain_scores = booster.get_score(importance_type="gain")
gain_series = pd.Series(gain_scores).reindex(
    [f"f{i}" for i in range(len(feat_names))], fill_value=0.0
)
# XGBoost uses f0, f1, ... internally — map back to real names
feat_idx_map = {f"f{i}": name for i, name in enumerate(feat_names)}
gain_named = {feat_idx_map.get(k, k): v for k, v in gain_scores.items()}
gain_series_named = pd.Series(gain_named).sort_values(ascending=False)

# (b) weight
weight_scores = booster.get_score(importance_type="weight")
weight_named = {feat_idx_map.get(k, k): v for k, v in weight_scores.items()}
weight_series_named = pd.Series(weight_named).sort_values(ascending=False)

# (c) permutation importance on test set
perm_result = permutation_importance(
    tuned_clf, X_te_enc, y_test,
    n_repeats=5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=RANDOM_SEED,
)
perm_series_named = pd.Series(
    perm_result.importances_mean, index=feat_names
).sort_values(ascending=False)

# Build comparison CSV — top 20 by gain, with weight rank and perm rank alongside
TOP_N_IMP = 20
top20_gain = gain_series_named.head(TOP_N_IMP)
gain_rank_map   = {feat: i+1 for i, feat in enumerate(gain_series_named.index)}
weight_rank_map = {feat: i+1 for i, feat in enumerate(weight_series_named.index)}
perm_rank_map   = {feat: i+1 for i, feat in enumerate(perm_series_named.index)}

imp_df = pd.DataFrame({
    "feature":     top20_gain.index,
    "gain_score":  top20_gain.values,
    "gain_rank":   [gain_rank_map[f] for f in top20_gain.index],
    "weight_rank": [weight_rank_map.get(f, 999) for f in top20_gain.index],
    "perm_rank":   [perm_rank_map.get(f, 999) for f in top20_gain.index],
    "perm_score":  [perm_series_named.get(f, 0.0) for f in top20_gain.index],
})
imp_df.to_csv(TABLES / "xgb_feature_importance.csv", index=False)
print(f"  Saved: {TABLES / 'xgb_feature_importance.csv'}")

top5_gain = list(zip(gain_series_named.head(5).index.tolist(),
                     [round(v, 6) for v in gain_series_named.head(5).values]))
top5_perm = list(zip(perm_series_named.head(5).index.tolist(),
                     [round(v, 6) for v in perm_series_named.head(5).values]))

print("\n  Top 5 Gain features:")
for i, (f, v) in enumerate(top5_gain, 1):
    print(f"    {i}. {f:<42} {v:.5f}")
print("\n  Top 5 Permutation features:")
for i, (f, v) in enumerate(top5_perm, 1):
    print(f"    {i}. {f:<42} {v:.6f}")

# ──────────────────────────────────────────────
# 8. Save xgb_results.json
# ──────────────────────────────────────────────
print("\n[7] Saving xgb_results.json...")

# Load comparison models for figures
import joblib as jl
lr_pipe = jl.load(MODELS / "logistic.pkl")
rf_pipe = jl.load(MODELS / "random_forest.pkl")

# LR/RF probabilities on test set — pass only the columns each pipeline expects
lr_feat_in = lr_pipe.named_steps["preprocessor"].feature_names_in_
rf_feat_in = rf_pipe.named_steps["preprocessor"].feature_names_in_

X_test_lr = test[list(lr_feat_in)]
X_test_rf = test[list(rf_feat_in)]

y_prob_lr = lr_pipe.predict_proba(X_test_lr)[:, 1]
y_prob_rf = rf_pipe.predict_proba(X_test_rf)[:, 1]

res_lr = evaluate(y_test, y_prob_lr)
res_rf = evaluate(y_test, y_prob_rf)

results_json = {
    "vanilla_metrics": {k: (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v)
                        for k, v in res_vanilla.items()},
    "tuned_metrics": {k: (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v)
                      for k, v in res_tuned.items()},
    "best_params": {k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, (int, np.integer)) else v)
                    for k, v in best_params.items()},
    "cv_auc": float(best_cv_auc),
    "n_trials_run": n_trials_run,
    "scale_pos_weight": float(scale_pos_weight),
    "best_iter_vanilla": int(best_iter_vanilla),
    "best_iter_tuned": int(best_iter_tuned),
    "top5_gain": [(str(f), float(v)) for f, v in top5_gain],
    "top5_perm": [(str(f), float(v)) for f, v in top5_perm],
    "auc_delta_vs_rf": float(res_tuned["auc"] - res_rf["auc"]),
    "auc_delta_vs_lr": float(res_tuned["auc"] - res_lr["auc"]),
    "lr_auc": float(res_lr["auc"]),
    "rf_auc": float(res_rf["auc"]),
}

with open(TABLES / "xgb_results.json", "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2)
print(f"  Saved: {TABLES / 'xgb_results.json'}")

# ──────────────────────────────────────────────
# 9. Save tuned pipeline
# ──────────────────────────────────────────────
print("\n[8] Saving tuned pipeline...")

# Build a proper sklearn pipeline with preprocessor
tuned_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", tuned_clf),
])
# The preprocessor was already fit; just wrap it
joblib.dump(tuned_pipe, MODELS / "xgboost.pkl")
# Smoke test
loaded_pipe = joblib.load(MODELS / "xgboost.pkl")
smoke = loaded_pipe.named_steps["classifier"].predict_proba(
    loaded_pipe.named_steps["preprocessor"].transform(X_test.head(5))
)[:, 1]
print(f"  Saved: {MODELS / 'xgboost.pkl'}")
print(f"  Smoke test (5 rows): {smoke.round(4).tolist()}")

# ──────────────────────────────────────────────
# 10. Figures
# ──────────────────────────────────────────────
print("\n[9] Generating figures...")

TOP_N_FIG = 20

# Figure 1: Gain importance
fig, ax = plt.subplots(figsize=(9, 8))
top_gain = gain_series_named.head(TOP_N_FIG).sort_values()
top_gain.plot(kind="barh", ax=ax, color="#2196F3")
ax.set_title(f"XGBoost Feature Importance (Gain) — Top {TOP_N_FIG}", fontsize=13)
ax.set_xlabel("Mean Gain per Split")
plt.tight_layout()
plt.savefig(FIG_DIR / "gain_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: gain_importance.png")

# Figure 2: Weight importance
top_weight = weight_series_named.head(TOP_N_FIG).sort_values()
fig, ax = plt.subplots(figsize=(9, 8))
top_weight.plot(kind="barh", ax=ax, color="#FF9800")
ax.set_title(f"XGBoost Feature Importance (Weight/Count) — Top {TOP_N_FIG}", fontsize=13)
ax.set_xlabel("Number of Times Feature Used in Splits")
plt.tight_layout()
plt.savefig(FIG_DIR / "weight_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: weight_importance.png")

# Figure 3: Permutation importance
top_perm = perm_series_named.head(TOP_N_FIG).sort_values()
fig, ax = plt.subplots(figsize=(9, 8))
top_perm.plot(kind="barh", ax=ax, color="#4CAF50")
ax.set_title(f"XGBoost Permutation Importance (Test Set) — Top {TOP_N_FIG}", fontsize=13)
ax.set_xlabel("Mean AUC Drop When Feature Shuffled")
plt.tight_layout()
plt.savefig(FIG_DIR / "permutation_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: permutation_importance.png")

# Figure 4: ROC/PR 4-curve comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
models_plot = {
    "Logistic_Optimized": (y_prob_lr, "#4C72B0"),
    "RF_Tuned":           (y_prob_rf, "#DD8452"),
    "XGB_Vanilla":        (y_prob_vanilla, "#55A868"),
    "XGB_Tuned":          (y_prob_tuned, "#C44E52"),
}
for name, (proba, color) in models_plot.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax1.plot(fpr, tpr, label=f"{name} ({sk_auc(fpr, tpr):.4f})", color=color, lw=2)
    prec_c, rec_c, _ = precision_recall_curve(y_test, proba)
    ax2.plot(rec_c, prec_c, label=f"{name} ({sk_auc(rec_c, prec_c):.4f})", color=color, lw=2)

ax1.plot([0, 1], [0, 1], "k--", lw=1)
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("ROC Curve — 4-Model Comparison")
ax1.legend(loc="lower right", fontsize=8)

ax2.axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"Baseline ({y_test.mean():.3f})")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve — 4-Model Comparison")
ax2.legend(loc="upper right", fontsize=8)

plt.suptitle("Phase 2.3 — XGBoost vs Prior Models", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_pr_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: roc_pr_comparison.png")

# Figure 5: Calibration curve — XGBoost Tuned
frac_pos_xgb,  mean_pred_xgb  = sk_cal_curve(y_test, y_prob_tuned, n_bins=10, strategy="uniform")
frac_pos_lr_c, mean_pred_lr_c = sk_cal_curve(y_test, y_prob_lr, n_bins=10, strategy="uniform")
frac_pos_rf_c, mean_pred_rf_c = sk_cal_curve(y_test, y_prob_rf, n_bins=10, strategy="uniform")

mace_xgb = float(np.mean(np.abs(frac_pos_xgb - mean_pred_xgb)))
mace_lr  = float(np.mean(np.abs(frac_pos_lr_c - mean_pred_lr_c)))
mace_rf  = float(np.mean(np.abs(frac_pos_rf_c - mean_pred_rf_c)))
print(f"\n  Calibration MACE — LR: {mace_lr:.4f}  RF: {mace_rf:.4f}  XGB_Tuned: {mace_xgb:.4f}")

# update results json with calibration
results_json["mace_xgb_tuned"] = mace_xgb
results_json["mace_lr"] = mace_lr
results_json["mace_rf"] = mace_rf
with open(TABLES / "xgb_results.json", "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.plot(mean_pred_lr_c, frac_pos_lr_c, "o-",
        label=f"LR (MACE={mace_lr:.3f})", color="#4C72B0")
ax.plot(mean_pred_rf_c, frac_pos_rf_c, "s-",
        label=f"RF_Tuned (MACE={mace_rf:.3f})", color="#DD8452")
ax.plot(mean_pred_xgb, frac_pos_xgb, "^-",
        label=f"XGB_Tuned (MACE={mace_xgb:.3f})", color="#C44E52", lw=2, ms=8)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve — LR vs RF vs XGB")
ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(FIG_DIR / "calibration_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: calibration_curve.png")

# Figure 6: Threshold analysis
thresholds = np.arange(0.10, 0.95, 0.05)
precisions_t, recalls_t, f1s_t = [], [], []
for thr in thresholds:
    yp = (y_prob_tuned >= thr).astype(int)
    precisions_t.append(precision_score(y_test, yp, zero_division=0))
    recalls_t.append(recall_score(y_test, yp, zero_division=0))
    f1s_t.append(f1_score(y_test, yp, zero_division=0))

precisions_t = np.array(precisions_t)
recalls_t    = np.array(recalls_t)
f1s_t        = np.array(f1s_t)

best_f1_idx = int(np.argmax(f1s_t))
best_f1_thr = float(thresholds[best_f1_idx])
rc_mask     = recalls_t >= 0.80
rc_thr = float(thresholds[rc_mask][np.argmax(precisions_t[rc_mask])]) if rc_mask.any() else None

print(f"\n  F1-optimal threshold:  {best_f1_thr:.2f}")
print(f"  Recall>=0.80 threshold: {rc_thr:.2f}" if rc_thr else "  Recall>=0.80: not achievable in range")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, precisions_t, label="Precision", color="#4C72B0", lw=2)
ax.plot(thresholds, recalls_t,    label="Recall",    color="#DD8452", lw=2)
ax.plot(thresholds, f1s_t,        label="F1 Score",  color="#55A868", lw=2)
ax.axvline(0.5, color="gray", ls="--", lw=1.2, label="Default (0.5)")
ax.axvline(best_f1_thr, color="#55A868", ls=":", lw=1.5, label=f"F1-optimal ({best_f1_thr:.2f})")
if rc_thr:
    ax.axvline(rc_thr, color="#DD8452", ls=":", lw=1.5, label=f"Recall>=0.8 ({rc_thr:.2f})")
ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
ax.set_title("Threshold Analysis — XGBoost_Tuned")
ax.set_xlim(0.10, 0.90); ax.set_ylim(0, 1)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: threshold_analysis.png")

# ──────────────────────────────────────────────
# 11. Final benchmark print
# ──────────────────────────────────────────────
print("\n[10] Final benchmark.csv:")
bench = pd.read_csv(BENCHMARK_CSV)
print(bench[["model", "auc", "pr_auc", "f1", "precision_at_k", "recall_at_k"]].to_string(index=False))

# Print summary for notebook embedding
print("\n" + "="*60)
print("SUMMARY FOR NOTEBOOK EMBEDDING")
print("="*60)
print(f"Vanilla AUC:          {res_vanilla['auc']:.4f}")
print(f"Tuned AUC:            {res_tuned['auc']:.4f}")
print(f"RF Tuned AUC:         {res_rf['auc']:.4f}")
print(f"LR Optimized AUC:     {res_lr['auc']:.4f}")
print(f"AUC delta vs RF:      {res_tuned['auc'] - res_rf['auc']:+.4f}")
print(f"AUC delta vs LR:      {res_tuned['auc'] - res_lr['auc']:+.4f}")
print(f"Best CV AUC (Optuna): {best_cv_auc:.4f}")
print(f"Best iter (tuned):    {best_iter_tuned}")
print(f"scale_pos_weight:     {scale_pos_weight:.4f}")
print(f"MACE XGB: {mace_xgb:.4f}  LR: {mace_lr:.4f}  RF: {mace_rf:.4f}")
print(f"F1-optimal threshold: {best_f1_thr:.2f}")
if rc_thr:
    print(f"Recall>=0.80 thresh:  {rc_thr:.2f}")
print(f"Top 5 Gain: {[f for f,v in top5_gain]}")
print(f"Top 5 Perm: {[f for f,v in top5_perm]}")
print("="*60)
print("\nrun_xgb.py complete.")
