# -*- coding: utf-8 -*-
"""
run_cluster_predict.py — Phase 2.6 Cluster-then-Predict
=========================================================
Stage 1: K-means clustering on behavioral features (train only)
Stage 2: Per-cluster XGBoost models (Phase 2.3 best params, no re-tuning)
Compares aggregated cluster-predict AUC vs global XGBoost on same frozen test.

Run from repo root:
    .venv\Scripts\python run_cluster_predict.py
Expected runtime: 30-50 min
"""
import os, sys, json, time, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc as sk_auc,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.config import (
    TRAIN_PATH, TEST_PATH, FIGURES, MODELS, TABLES,
    BENCHMARK_CSV, RANDOM_SEED
)
from src.evaluate import evaluate, print_metrics

# ──────────────────────────────────────────────
# 0. Output dirs
# ──────────────────────────────────────────────
FIG_DIR = FIGURES / "06_cluster_predict"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"[setup] Figures -> {FIG_DIR}")

# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
print("\n[1] Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

TARGET   = "churned"
METADATA = ["wallet_id", "full_name", "churned_vendor"]

ALL_FEATURE_COLS = [c for c in train.columns if c not in [TARGET] + METADATA]
CATEGORICAL_COLS = [
    "gender", "state", "city", "referral_source",
    "preferred_language", "linked_bank", "kyc_tier",
]
NUMERIC_COLS = [c for c in ALL_FEATURE_COLS if c not in CATEGORICAL_COLS]

# Clustering uses a focused behavioral subset — avoid noisy high-cardinality features
CLUSTER_FEATURES = [
    "txn_count", "total_amount_ngn", "avg_amount_ngn",
    "days_since_last_txn", "account_active_span_days",
    "txn_velocity_change", "txn_per_active_day",
    "unique_channels", "unique_txn_types",
    "is_reactivator", "dormancy_days_before_reactivation",
    "tenure_days", "fraud_rate",
]
# Validate all are present
missing_cf = [f for f in CLUSTER_FEATURES if f not in train.columns]
if missing_cf:
    raise ValueError(f"Missing clustering features: {missing_cf}")

y_train = train[TARGET].values
y_test  = test[TARGET].values

print(f"  Train: {train.shape}  |  Test: {test.shape}")
print(f"  Churn rate — train: {y_train.mean():.4f}  test: {y_test.mean():.4f}")
print(f"  Clustering features: {len(CLUSTER_FEATURES)}")

# ──────────────────────────────────────────────
# 2. Preprocessors
# ──────────────────────────────────────────────
print("\n[2] Building preprocessors...")

# Clustering: StandardScaler on the 13 behavioral features (fit train only)
cluster_scaler = StandardScaler()
X_train_cluster = cluster_scaler.fit_transform(train[CLUSTER_FEATURES].values)
X_test_cluster  = cluster_scaler.transform(test[CLUSTER_FEATURES].values)
print(f"  Cluster scaler fit OK (train shape: {X_train_cluster.shape})")

# XGBoost: OrdinalEncoder for categoricals — same as Phase 2.3
# Load from saved xgboost.pkl (shared preprocessor, fit on full train)
global_pipe = joblib.load(MODELS / "xgboost.pkl")
xgb_preprocessor = global_pipe.named_steps["preprocessor"]
X_train_enc = xgb_preprocessor.transform(train[ALL_FEATURE_COLS])
X_test_enc  = xgb_preprocessor.transform(test[ALL_FEATURE_COLS])
feat_names  = xgb_preprocessor.get_feature_names_out()
print(f"  XGBoost preprocessor loaded from xgboost.pkl ({len(feat_names)} features)")

# ──────────────────────────────────────────────
# 3. K selection — elbow + silhouette
# ──────────────────────────────────────────────
print("\n[3] K-means K selection (K=2..6)...")

K_RANGE = [2, 3, 4, 5, 6]
inertias    = []
sil_scores  = []

# Sample for silhouette (full train is 300K — silhouette on 300K is slow)
# Use a stratified 10% sample
rng = np.random.default_rng(RANDOM_SEED)
sil_sample_idx = rng.choice(len(X_train_cluster), size=min(30000, len(X_train_cluster)), replace=False)
X_sil = X_train_cluster[sil_sample_idx]

for k in K_RANGE:
    t0 = time.time()
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels_full = km.fit_predict(X_train_cluster)
    inertias.append(km.inertia_)
    # Silhouette on sample
    labels_sample = labels_full[sil_sample_idx]
    sil = silhouette_score(X_sil, labels_sample, sample_size=None)
    sil_scores.append(sil)
    print(f"  K={k}: inertia={km.inertia_:.0f}  silhouette={sil:.4f}  ({time.time()-t0:.1f}s)")

# Auto-select K: pick highest silhouette among K>=3; fall back to 4 if margin < 0.02
best_k_idx = int(np.argmax(sil_scores[1:]))  # index into K_RANGE[1:] (K>=3 starts at idx 1)
best_k_candidate = K_RANGE[1:][best_k_idx]
best_sil = sil_scores[1:][best_k_idx]
k4_sil   = sil_scores[K_RANGE.index(4)]

if abs(best_sil - k4_sil) < 0.02:
    CHOSEN_K = 4
    print(f"\n  Silhouette margin < 0.02 between K={best_k_candidate} ({best_sil:.4f}) and K=4 ({k4_sil:.4f}) — defaulting to K=4 for interpretability")
else:
    CHOSEN_K = best_k_candidate
    print(f"\n  Auto-selected K={CHOSEN_K} (silhouette={best_sil:.4f})")

# K-selection figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(K_RANGE, inertias, "o-", color="#4C72B0", lw=2)
ax1.set_xlabel("K"); ax1.set_ylabel("Inertia (WCSS)")
ax1.set_title("Elbow Curve — K-means Inertia")
ax1.set_xticks(K_RANGE)
ax2.plot(K_RANGE, sil_scores, "s-", color="#DD8452", lw=2)
ax2.axvline(CHOSEN_K, color="gray", ls="--", lw=1.5, label=f"Chosen K={CHOSEN_K}")
ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score by K")
ax2.set_xticks(K_RANGE); ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "kmeans_k_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: kmeans_k_selection.png")

# ──────────────────────────────────────────────
# 4. Fit chosen K-means, assign clusters
# ──────────────────────────────────────────────
print(f"\n[4] Fitting K={CHOSEN_K} K-means on full training set...")
t0 = time.time()
kmeans = KMeans(n_clusters=CHOSEN_K, random_state=RANDOM_SEED, n_init=20)
train_clusters = kmeans.fit_predict(X_train_cluster)
test_clusters  = kmeans.predict(X_test_cluster)
joblib.dump(kmeans, MODELS / "kmeans.pkl")
print(f"  Done in {time.time()-t0:.1f}s. Saved: kmeans.pkl")
print(f"  Train cluster distribution: {np.bincount(train_clusters).tolist()}")
print(f"  Test  cluster distribution: {np.bincount(test_clusters).tolist()}")

# ──────────────────────────────────────────────
# 5. Cluster profiling
# ──────────────────────────────────────────────
print("\n[5] Profiling clusters...")

profile_rows = []
for k in range(CHOSEN_K):
    mask_tr = train_clusters == k
    mask_te = test_clusters  == k
    df_k    = train[mask_tr]
    n_tr    = int(mask_tr.sum())
    n_te    = int(mask_te.sum())
    n_churn = int(y_train[mask_tr].sum())
    churn_r = float(y_train[mask_tr].mean())
    row = {
        "cluster": k,
        "n_train": n_tr,
        "n_test":  n_te,
        "pct_train": round(n_tr / len(train) * 100, 2),
        "n_churners_train": n_churn,
        "churn_rate": round(churn_r, 4),
    }
    for feat in CLUSTER_FEATURES:
        row[f"mean_{feat}"] = round(float(df_k[feat].mean()), 4)
    profile_rows.append(row)

profiles_df = pd.DataFrame(profile_rows)
profiles_df.to_csv(TABLES / "cluster_profiles.csv", index=False)
print(f"  Saved: cluster_profiles.csv")
print("\n  Cluster summary:")
summary_cols = ["cluster","n_train","pct_train","n_churners_train","churn_rate",
                "mean_txn_count","mean_days_since_last_txn",
                "mean_is_reactivator","mean_dormancy_days_before_reactivation"]
print(profiles_df[summary_cols].to_string(index=False))

# ──────────────────────────────────────────────
# 6. Load Phase 2.3 best params
# ──────────────────────────────────────────────
print("\n[6] Loading Phase 2.3 best params...")
with open(TABLES / "xgb_results.json", encoding="utf-8") as f:
    xgb_results = json.load(f)

best_params   = xgb_results["best_params"]
best_iter     = xgb_results["best_iter_tuned"]   # 249 — use as fixed n_estimators
print(f"  best_params loaded: n_estimators={best_params['n_estimators']}, "
      f"max_depth={best_params['max_depth']}, lr={best_params['learning_rate']:.5f}")
print(f"  Using fixed n_estimators={best_iter} (global model best_iteration)")

# ──────────────────────────────────────────────
# 7. Per-cluster XGBoost training
# ──────────────────────────────────────────────
print("\n[7] Training per-cluster XGBoost models...")

MIN_CLUSTER_SIZE   = 1000
MIN_CLUSTER_CHURN  = 100

cluster_models     = {}   # k -> fitted clf (or None if fallback to global)
cluster_fallback   = {}   # k -> bool

# Load global clf for fallback
global_clf = global_pipe.named_steps["classifier"]

feat_idx_map = {f"f{i}": name for i, name in enumerate(feat_names)}

for k in range(CHOSEN_K):
    mask_tr = train_clusters == k
    n_tr    = int(mask_tr.sum())
    n_churn = int(y_train[mask_tr].sum())
    print(f"\n  Cluster {k}: n_train={n_tr}, churners={n_churn}, "
          f"churn_rate={n_churn/n_tr:.4f}")

    # Pure-class cluster: XGBoost cannot train binary classifier on single class
    is_pure = (n_churn == n_tr or n_churn == 0)
    if n_tr < MIN_CLUSTER_SIZE or n_churn < MIN_CLUSTER_CHURN or is_pure:
        reason = ("pure-class (all churned)" if n_churn == n_tr
                  else "pure-class (no churners)" if n_churn == 0
                  else "below size threshold")
        print(f"    -> FALLBACK to global model ({reason})")
        cluster_models[k]   = None
        cluster_fallback[k] = True
        continue

    cluster_fallback[k] = False
    X_k = X_train_enc[mask_tr]
    y_k = y_train[mask_tr]

    n_neg_k = int((y_k == 0).sum())
    n_pos_k = int((y_k == 1).sum())
    spw_k   = n_neg_k / n_pos_k
    print(f"    scale_pos_weight={spw_k:.3f}")

    cluster_params = {
        **best_params,
        "n_estimators":      best_iter,    # fixed at global best_iteration
        "scale_pos_weight":  spw_k,
        "tree_method":       "hist",
        "eval_metric":       "auc",
        "n_jobs":            -1,
        "random_state":      RANDOM_SEED,
        "verbosity":         0,
    }
    # Remove early_stopping-related params that only work with eval_set
    cluster_params.pop("early_stopping_rounds", None)

    t0 = time.time()
    clf_k = xgb.XGBClassifier(**cluster_params)
    clf_k.fit(X_k, y_k, verbose=False)
    elapsed = time.time() - t0

    # Quick in-sample sanity check
    proba_in = clf_k.predict_proba(X_k)[:, 1]
    auc_in   = roc_auc_score(y_k, proba_in)
    print(f"    Trained in {elapsed:.1f}s | in-sample AUC={auc_in:.4f}")

    cluster_models[k] = clf_k

    # Save as pipeline
    pipe_k = Pipeline([
        ("preprocessor", xgb_preprocessor),
        ("classifier",   clf_k),
    ])
    joblib.dump(pipe_k, MODELS / f"xgb_cluster_{k}.pkl")
    print(f"    Saved: xgb_cluster_{k}.pkl")

# ──────────────────────────────────────────────
# 8. Evaluation
# ──────────────────────────────────────────────
print("\n[8] Evaluating...")

# (a) Global model predictions on test
y_prob_global = global_clf.predict_proba(X_test_enc)[:, 1]
auc_global    = roc_auc_score(y_test, y_prob_global)
print(f"  Global XGB AUC on full test: {auc_global:.4f}")

# (b) Cluster-routed predictions
y_prob_cluster = np.zeros(len(y_test))
for k in range(CHOSEN_K):
    mask_te = test_clusters == k
    if not mask_te.any():
        continue
    if cluster_fallback[k]:
        proba_k = global_clf.predict_proba(X_test_enc[mask_te])[:, 1]
    else:
        proba_k = cluster_models[k].predict_proba(X_test_enc[mask_te])[:, 1]
    y_prob_cluster[mask_te] = proba_k

res_cluster = evaluate(y_test, y_prob_cluster)
print_metrics(res_cluster, "ClusterPredict_XGB")

# (c) Per-cluster detailed comparison
print("\n  Per-cluster breakdown:")
per_cluster_rows = []
for k in range(CHOSEN_K):
    mask_te   = test_clusters == k
    n_te_k    = int(mask_te.sum())
    y_te_k    = y_test[mask_te]
    n_churn_k = int(y_te_k.sum())
    churn_r_k = float(y_te_k.mean())

    if n_te_k == 0 or n_churn_k == 0:
        print(f"  Cluster {k}: no test data or no churners — skip AUC")
        per_cluster_rows.append({
            "cluster": k,
            "n_test": n_te_k, "n_churners": n_churn_k,
            "churn_rate": churn_r_k,
            "auc_cluster_model": None, "auc_global_restricted": None,
            "lift": None, "fallback": cluster_fallback[k],
        })
        continue

    # Cluster-model AUC on this slice
    if cluster_fallback[k]:
        proba_slice_cm = global_clf.predict_proba(X_test_enc[mask_te])[:, 1]
    else:
        proba_slice_cm = cluster_models[k].predict_proba(X_test_enc[mask_te])[:, 1]
    auc_cm = roc_auc_score(y_te_k, proba_slice_cm)

    # Global model restricted to this cluster's test slice
    proba_slice_gl = global_clf.predict_proba(X_test_enc[mask_te])[:, 1]
    auc_gl_restricted = roc_auc_score(y_te_k, proba_slice_gl)

    lift = auc_cm - auc_gl_restricted

    print(f"  Cluster {k}: n_test={n_te_k} ({churn_r_k:.2%} churn) | "
          f"AUC cluster={auc_cm:.4f}  global={auc_gl_restricted:.4f}  "
          f"lift={lift:+.4f}  fallback={cluster_fallback[k]}")

    per_cluster_rows.append({
        "cluster": k,
        "n_test": n_te_k, "n_churners": n_churn_k,
        "churn_rate": round(churn_r_k, 4),
        "auc_cluster_model": round(auc_cm, 4),
        "auc_global_restricted": round(auc_gl_restricted, 4),
        "lift": round(lift, 4),
        "fallback": cluster_fallback[k],
    })

per_cluster_df = pd.DataFrame(per_cluster_rows)
per_cluster_df.to_csv(TABLES / "per_cluster_results.csv", index=False)
print(f"\n  Saved: per_cluster_results.csv")

# ──────────────────────────────────────────────
# 9. safe_append to benchmark
# ──────────────────────────────────────────────
def safe_append(model_name: str, results: dict, notes: str = "") -> None:
    row = {"model": model_name, **results, "notes": notes}
    df_new = pd.DataFrame([row])
    if BENCHMARK_CSV.exists():
        df = pd.read_csv(BENCHMARK_CSV)
        if model_name in df["model"].values:
            print(f"[benchmark] {model_name} already present — skipping.")
            return
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(BENCHMARK_CSV, index=False)
    print(f"[benchmark] appended {model_name}")

fallback_clusters = [k for k in range(CHOSEN_K) if cluster_fallback[k]]
trained_clusters  = [k for k in range(CHOSEN_K) if not cluster_fallback[k]]
notes_cluster = (
    f"K={CHOSEN_K} KMeans; {len(trained_clusters)} cluster-specific models "
    f"(fallback to global: {fallback_clusters}); "
    f"XGB params reused from Phase 2.3 (no re-tuning); "
    f"n_est={best_iter} fixed"
)
safe_append("ClusterPredict_XGB", res_cluster, notes=notes_cluster)

# ──────────────────────────────────────────────
# 10. Feature importance heatmap
# ──────────────────────────────────────────────
print("\n[9] Building feature importance heatmap...")
TOP_N_HEAT = 15

# Get global model gain for top features (reference ranking)
global_booster = global_clf.get_booster()
global_gain = global_booster.get_score(importance_type="gain")
global_gain_named = {feat_idx_map.get(k, k): v for k, v in global_gain.items()}
global_gain_series = pd.Series(global_gain_named).sort_values(ascending=False)
top_features = list(global_gain_series.head(TOP_N_HEAT).index)

heat_data = {}
# Global model column
heat_data["Global_XGB"] = {f: global_gain_named.get(f, 0.0) for f in top_features}

for k in range(CHOSEN_K):
    if cluster_fallback[k]:
        col_name = f"Cluster_{k}\n(fallback)"
        heat_data[col_name] = {f: global_gain_named.get(f, 0.0) for f in top_features}
    else:
        booster_k = cluster_models[k].get_booster()
        gain_k    = booster_k.get_score(importance_type="gain")
        gain_named_k = {feat_idx_map.get(fk, fk): v for fk, v in gain_k.items()}
        col_name = f"Cluster_{k}\n(n={int((train_clusters==k).sum()):,})"
        heat_data[col_name] = {f: gain_named_k.get(f, 0.0) for f in top_features}

heat_df = pd.DataFrame(heat_data, index=top_features)
# Normalize each column to 0-1 for comparable display
heat_norm = heat_df.div(heat_df.max(axis=0), axis=1).fillna(0.0)

fig, ax = plt.subplots(figsize=(max(8, CHOSEN_K*2 + 3), 7))
sns.heatmap(
    heat_norm, annot=False, cmap="YlOrRd",
    linewidths=0.5, ax=ax, cbar_kws={"label": "Normalized gain (0-1 per model)"}
)
ax.set_title(f"Feature Importance Heatmap — Global XGB vs K={CHOSEN_K} Cluster Models\n"
             f"(Gain, normalized per model)", fontsize=11)
ax.set_ylabel("Feature"); ax.set_xlabel("Model")
plt.tight_layout()
plt.savefig(FIG_DIR / "cluster_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: cluster_feature_importance.png")

# ──────────────────────────────────────────────
# 11. Per-cluster lift figure
# ──────────────────────────────────────────────
print("\n[10] Generating per_cluster_lift.png...")

valid_rows = per_cluster_df.dropna(subset=["auc_cluster_model"])
x    = np.arange(len(valid_rows))
w    = 0.35
fig, ax = plt.subplots(figsize=(max(8, len(valid_rows)*2), 5))
bars1 = ax.bar(x - w/2, valid_rows["auc_cluster_model"],    w, label="Cluster-specific model", color="#C44E52")
bars2 = ax.bar(x + w/2, valid_rows["auc_global_restricted"], w, label="Global XGB (same slice)", color="#4C72B0", alpha=0.7)
ax.axhline(auc_global, color="gray", ls="--", lw=1.5, label=f"Global XGB full test ({auc_global:.4f})")
ax.set_xlabel("Cluster")
ax.set_ylabel("AUC")
ax.set_title("Per-Cluster AUC: Cluster-Specific vs Global XGB (restricted to cluster slice)")
ax.set_xticks(x)
churn_labels = [f"Cluster {r['cluster']}\n({r['churn_rate']:.1%} churn)"
                for _, r in valid_rows.iterrows()]
ax.set_xticklabels(churn_labels)
ax.legend(fontsize=9); ax.set_ylim(0.4, 1.02)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "per_cluster_lift.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: per_cluster_lift.png")

# ──────────────────────────────────────────────
# 12. ROC/PR 3-curve comparison
# ──────────────────────────────────────────────
print("\n[11] Generating roc_pr_comparison.png...")

rf_pipe  = joblib.load(MODELS / "random_forest.pkl")
rf_feat  = rf_pipe.named_steps["preprocessor"].feature_names_in_
y_prob_rf = rf_pipe.predict_proba(test[list(rf_feat)])[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
models_plot = {
    "Global_XGB":       (y_prob_global,  "#4C72B0"),
    "RF_Tuned":         (y_prob_rf,      "#DD8452"),
    "ClusterPredict":   (y_prob_cluster, "#C44E52"),
}
for name, (proba, color) in models_plot.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax1.plot(fpr, tpr, label=f"{name} ({sk_auc(fpr,tpr):.4f})", color=color, lw=2)
    prec_c, rec_c, _ = precision_recall_curve(y_test, proba)
    ax2.plot(rec_c, prec_c, label=f"{name} ({sk_auc(rec_c,prec_c):.4f})", color=color, lw=2)

ax1.plot([0,1],[0,1],"k--",lw=1)
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("ROC — Global XGB vs RF vs ClusterPredict")
ax1.legend(loc="lower right", fontsize=8)

ax2.axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"Baseline ({y_test.mean():.3f})")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("PR — Global XGB vs RF vs ClusterPredict")
ax2.legend(loc="upper right", fontsize=8)

plt.suptitle(f"Phase 2.6 — ClusterPredict (K={CHOSEN_K}) vs Prior Models", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_pr_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: roc_pr_comparison.png")

# ──────────────────────────────────────────────
# 13. Save cluster_predict_results.json
# ──────────────────────────────────────────────
print("\n[12] Saving cluster_predict_results.json...")

results_out = {
    "chosen_k":         CHOSEN_K,
    "k_range":          K_RANGE,
    "inertias":         [float(x) for x in inertias],
    "silhouette_scores":[float(x) for x in sil_scores],
    "cluster_sizes_train": [int(x) for x in np.bincount(train_clusters)],
    "cluster_sizes_test":  [int(x) for x in np.bincount(test_clusters)],
    "cluster_churn_rates": [
        round(float(y_train[train_clusters==k].mean()), 4)
        for k in range(CHOSEN_K)
    ],
    "fallback_clusters": fallback_clusters,
    "trained_clusters":  trained_clusters,
    "global_xgb_auc":   round(auc_global, 6),
    "cluster_predict_auc": round(res_cluster["auc"], 6),
    "cluster_predict_pr_auc": round(res_cluster["pr_auc"], 6),
    "auc_delta_vs_global": round(res_cluster["auc"] - auc_global, 6),
    "per_cluster": per_cluster_df.to_dict(orient="records"),
    "best_params_used":  best_params,
    "n_estimators_used": best_iter,
    "all_metrics": {k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v)
                    for k, v in res_cluster.items()},
}

with open(TABLES / "cluster_predict_results.json", "w", encoding="utf-8") as f:
    json.dump(results_out, f, indent=2)
print(f"  Saved: cluster_predict_results.json")

# ──────────────────────────────────────────────
# 14. Final benchmark + summary
# ──────────────────────────────────────────────
print("\n[13] Final benchmark.csv:")
bench = pd.read_csv(BENCHMARK_CSV)
print(bench[["model","auc","pr_auc","f1","precision_at_k","recall_at_k"]].to_string(index=False))

print("\n" + "="*70)
print("SUMMARY FOR NOTEBOOK EMBEDDING")
print("="*70)
print(f"Chosen K:                 {CHOSEN_K}")
print(f"Silhouette scores (K2-6): {[round(s,4) for s in sil_scores]}")
print(f"Trained cluster models:   {trained_clusters}")
print(f"Fallback clusters:        {fallback_clusters}")
print(f"Global XGB AUC:           {auc_global:.4f}")
print(f"ClusterPredict AUC:       {res_cluster['auc']:.4f}")
print(f"ClusterPredict PR-AUC:    {res_cluster['pr_auc']:.4f}")
print(f"AUC delta vs global:      {res_cluster['auc']-auc_global:+.4f}")
print(f"Ceiling broken (>0.760):  {res_cluster['auc'] > 0.760}")
print("\nPer-cluster breakdown:")
print(per_cluster_df[["cluster","n_test","churn_rate","auc_cluster_model",
                       "auc_global_restricted","lift","fallback"]].to_string(index=False))
print("="*70)
print("\nrun_cluster_predict.py complete.")
