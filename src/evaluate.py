"""
Canonical evaluation. Every model in every notebook uses this.

Do NOT compute metrics inline in notebooks. Always call evaluate().
This is what guarantees the benchmark.csv is comparable across models.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, log_loss, brier_score_loss,
    confusion_matrix,
)

from src.config import BENCHMARK_CSV


def evaluate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    top_k: float = 0.10,
) -> dict:
    """Return a dict of metrics for a binary classifier.

    Args:
        y_true: (n,) array of 0/1 true labels.
        y_pred_proba: (n,) array of predicted P(churn=1).
        threshold: decision threshold for hard-label metrics.
        top_k: fraction of highest-risk customers for precision@k / recall@k.
            Default 0.10 = top decile.

    Returns:
        dict with auc, pr_auc, f1, precision, recall, log_loss, brier,
        precision_at_k, recall_at_k, threshold, top_k, tn, fp, fn, tp.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Top-k metrics: customers ranked by predicted churn probability
    n = len(y_true)
    k = max(1, int(np.ceil(top_k * n)))
    topk_idx = np.argsort(-y_pred_proba)[:k]
    precision_at_k = y_true[topk_idx].mean()
    recall_at_k = y_true[topk_idx].sum() / max(y_true.sum(), 1)

    return {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "log_loss": log_loss(y_true, np.clip(y_pred_proba, 1e-7, 1 - 1e-7)),
        "brier": brier_score_loss(y_true, y_pred_proba),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "threshold": threshold,
        "top_k": top_k,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def append_to_benchmark(model_name: str, results: dict, notes: str = "") -> None:
    """Append a single model's results to outputs/tables/benchmark.csv.

    One row per model. Call this from every modeling notebook.
    """
    row = {"model": model_name, **results, "notes": notes}
    df_new = pd.DataFrame([row])
    if BENCHMARK_CSV.exists():
        df = pd.read_csv(BENCHMARK_CSV)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(BENCHMARK_CSV, index=False)
    print(f"[benchmark] appended {model_name} → {BENCHMARK_CSV}")


def print_metrics(results: dict, model_name: str = "model") -> None:
    """Pretty-print evaluation results."""
    print(f"\n=== {model_name} ===")
    for k in ["auc", "pr_auc", "f1", "precision", "recall",
              "precision_at_k", "recall_at_k", "log_loss", "brier"]:
        v = results.get(k)
        if v is not None:
            print(f"  {k:>16}: {v:.4f}")
    print(f"  confusion: TN={results['tn']:,}  FP={results['fp']:,}  "
          f"FN={results['fn']:,}  TP={results['tp']:,}")
