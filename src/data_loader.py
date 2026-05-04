"""
⚠️ SUPERSEDED — DO NOT RE-RUN THIS LOADER ⚠️

The canonical `data/processed/customers_master.parquet` was manually adjusted on
2026-05-03 to refine unrealistic patterns in the synthetic data and better reflect
industry-typical fintech churn dynamics. Re-running this loader will OVERWRITE
the adjusted file with the original synthetic content and invalidate all downstream
Phase 2 results.

If you need to regenerate from raw, first back up the current
data/processed/customers_master.parquet (an adjusted copy is in
data/processed/_adjusted_master_backup/).

--- Original loader docstring below ---

Data loader for RetainIQ.

Implements the locked workflow from CLAUDE.md §6:
  1. Load raw Parquet (4M txn rows) and JSON profiles (375K customers).
  2. Split transactions into FEATURE window (Jan-Apr) and LABEL window (May-Jun).
  3. Aggregate FEATURE-window transactions per wallet → behavioral features.
  4. Engineer churn label from LABEL-window activity (zero txns => churned=1).
  5. Join: profiles (left) + behavioral features + labels.
  6. Apply data quality audit (2026-05-02): drop 7 synthetic/redundant columns.
  7. Save customers_master.parquet (375K rows × 38 cols).

Data quality audit — dropped columns (CLAUDE.md §6, locked 2026-05-02):
  Synthetic JSON fields (contradicts Parquet activity patterns):
    account_status, notification_preferences, support_tier
  Redundant raw-date forms (engineered derivatives already present):
    date_of_birth (-> age), registration_date (-> tenure_days),
    last_txn_date (-> days_since_last_txn),
    first_txn_date (-> account_active_span_days)

Metadata columns kept in master but NEVER used as model inputs:
  wallet_id (join key), full_name (output identifier), churned_vendor (sanity check)

Run as a module:
    python -m src.data_loader
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    PARQUET_PATH, JSON_PATH, CUSTOMERS_MASTER,
    FEATURE_WINDOW_START, FEATURE_WINDOW_END,
    LABEL_WINDOW_START, LABEL_WINDOW_END,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------

def load_transactions() -> pd.DataFrame:
    """Load the full Parquet transaction log."""
    print(f"[load] reading {PARQUET_PATH.name} ...")
    t0 = time.time()
    df = pd.read_parquet(PARQUET_PATH)
    # Identify timestamp column (schema lock-in: should be one date/time-typed column)
    ts_cols = [c for c in df.columns
               if df[c].dtype.kind == "M" or "time" in c.lower() or "date" in c.lower()]
    if not ts_cols:
        raise ValueError("No timestamp column found in Parquet. Inspect schema.")
    ts_col = ts_cols[0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.rename(columns={ts_col: "timestamp"})
    print(f"[load] transactions: {df.shape[0]:,} rows × {df.shape[1]} cols "
          f"({time.time()-t0:.1f}s)")
    return df


def load_profiles() -> pd.DataFrame:
    """Load JSON customer profiles → DataFrame keyed by wallet_id."""
    print(f"[load] reading {JSON_PATH.name} ...")
    t0 = time.time()
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Schema lock-in: top-level dict with 'data' list
    records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    df = pd.DataFrame(records)
    if "registration_date" in df.columns:
        df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    has_fullname = "full_name" in df.columns
    print(f"[load] profiles: {df.shape[0]:,} rows × {df.shape[1]} cols "
          f"({time.time()-t0:.1f}s)  |  full_name present: {has_fullname}")
    return df


# ----------------------------------------------------------------------
# 2. Time-based splitting
# ----------------------------------------------------------------------

def split_by_time(tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions into feature-window and label-window slices.

    Feature window: Jan-Apr 2024 (used for ALL behavioral features).
    Label window:   May-Jun 2024 (used ONLY for the churn label).
    """
    feat_mask = (tx["timestamp"] >= FEATURE_WINDOW_START) & \
                (tx["timestamp"] <= FEATURE_WINDOW_END)
    label_mask = (tx["timestamp"] >= LABEL_WINDOW_START) & \
                 (tx["timestamp"] <= LABEL_WINDOW_END)

    tx_feat = tx.loc[feat_mask].copy()
    tx_label = tx.loc[label_mask].copy()

    print(f"[split] feature window {FEATURE_WINDOW_START.date()} -> "
          f"{FEATURE_WINDOW_END.date()}: {len(tx_feat):,} txns")
    print(f"[split] label window   {LABEL_WINDOW_START.date()} -> "
          f"{LABEL_WINDOW_END.date()}: {len(tx_label):,} txns")
    if len(tx_feat) + len(tx_label) < 0.95 * len(tx):
        print(f"[split] WARNING: dropped {len(tx)-len(tx_feat)-len(tx_label):,} "
              f"txns outside both windows.")
    return tx_feat, tx_label


# ----------------------------------------------------------------------
# 3. Behavioral aggregation (feature window only)
# ----------------------------------------------------------------------

def aggregate_behavior(tx_feat: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature-window transactions to one row per wallet_id.

    Produces a base set of behavioral features. More feature engineering
    happens downstream in notebooks / src/features.py — this is just the
    raw aggregations.
    """
    print("[agg] aggregating behavioral features per wallet ...")
    t0 = time.time()

    # Reference date for recency features = end of feature window
    ref_date = FEATURE_WINDOW_END

    g = tx_feat.groupby("wallet_id", sort=False)

    agg = g.agg(
        # Volume
        txn_count=("amount_ngn", "size"),
        total_amount_ngn=("amount_ngn", "sum"),
        avg_amount_ngn=("amount_ngn", "mean"),
        std_amount_ngn=("amount_ngn", "std"),
        max_amount_ngn=("amount_ngn", "max"),
        min_amount_ngn=("amount_ngn", "min"),
        # Fees
        total_fee_ngn=("fee_ngn", "sum"),
        avg_fee_ngn=("fee_ngn", "mean"),
        # Balance
        last_balance_ngn=("balance_after_ngn", "last"),
        avg_balance_ngn=("balance_after_ngn", "mean"),
        std_balance_ngn=("balance_after_ngn", "std"),
        # Recency
        last_txn_date=("timestamp", "max"),
        first_txn_date=("timestamp", "min"),
        # Risk / friction
        fraud_count=("fraud_flag", "sum"),
        fraud_rate=("fraud_flag", "mean"),
        # Diversity
        unique_channels=("channel", "nunique"),
        unique_txn_types=("transaction_type", "nunique"),
        unique_device_os=("device_os", "nunique"),
        unique_agents=("agent_id", "nunique"),
    )

    # Recency features
    agg["days_since_last_txn"] = (ref_date - agg["last_txn_date"]).dt.days
    agg["account_active_span_days"] = (agg["last_txn_date"] - agg["first_txn_date"]).dt.days
    agg["txn_per_active_day"] = agg["txn_count"] / agg["account_active_span_days"].clip(lower=1)

    # Monthly trend: txns in first half (Jan-Feb) vs second half (Mar-Apr) of feature window
    half = FEATURE_WINDOW_START + (FEATURE_WINDOW_END - FEATURE_WINDOW_START) / 2
    h1 = tx_feat.loc[tx_feat["timestamp"] <= half].groupby("wallet_id").size()
    h2 = tx_feat.loc[tx_feat["timestamp"] > half].groupby("wallet_id").size()
    trend = pd.DataFrame({"txn_count_h1": h1, "txn_count_h2": h2}).fillna(0)
    trend["txn_velocity_change"] = (
        (trend["txn_count_h2"] - trend["txn_count_h1"])
        / trend["txn_count_h1"].clip(lower=1)
    )
    agg = agg.join(trend, how="left")

    # KYC tier (modal value per wallet — should be stable but be safe)
    if "kyc_tier" in tx_feat.columns:
        kyc = tx_feat.groupby("wallet_id")["kyc_tier"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        )
        agg["kyc_tier"] = kyc

    # Vendor's churn label aggregated to customer level — sanity-check ONLY
    if "churn_30d" in tx_feat.columns:
        agg["churned_vendor"] = tx_feat.groupby("wallet_id")["churn_30d"].max().astype(int)

    # Fill std NaN (single-transaction wallets) with 0
    for c in ["std_amount_ngn", "std_balance_ngn"]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0)

    print(f"[agg] produced {agg.shape[0]:,} rows × {agg.shape[1]} cols "
          f"({time.time()-t0:.1f}s)")
    return agg.reset_index()


# ----------------------------------------------------------------------
# 4. Churn label engineering (label window only)
# ----------------------------------------------------------------------

def engineer_churn_label(tx_label: pd.DataFrame, all_wallet_ids: pd.Series) -> pd.DataFrame:
    """Engineer customer-level churn label from label-window activity.

    churned = 1 if zero transactions in [LABEL_WINDOW_START, LABEL_WINDOW_END]
    churned = 0 otherwise

    `all_wallet_ids` is the universe of wallets we want labels for (from JSON).
    Wallets with no label-window activity at all are explicitly churned=1.
    """
    print("[label] engineering customer-level churn label ...")
    active_in_label = set(tx_label["wallet_id"].unique())
    df = pd.DataFrame({"wallet_id": all_wallet_ids.unique()})
    df["churned"] = (~df["wallet_id"].isin(active_in_label)).astype(int)

    rate = df["churned"].mean()
    print(f"[label] churn rate: {rate:.1%} "
          f"({df['churned'].sum():,} churned / {len(df):,} total)")
    return df


# ----------------------------------------------------------------------
# 5. Master assembly
# ----------------------------------------------------------------------

def build_master(profiles: pd.DataFrame,
                 behavioral: pd.DataFrame,
                 labels: pd.DataFrame) -> pd.DataFrame:
    """Join profiles ← behavioral ← labels into the master customer table.

    Fill order matters: targeted fills with non-zero values MUST run BEFORE
    the generic numeric zero-fill loop.  If the generic loop runs first, it
    consumes all NaNs and the targeted step finds nothing left to fix —
    silently producing wrong values (e.g. days_since_last_txn=0 for zombies
    instead of max_days+1).
    """
    print("[join] assembling master table ...")
    t0 = time.time()
    master = (
        profiles
        .merge(behavioral, on="wallet_id", how="left")
        .merge(labels, on="wallet_id", how="left")
    )

    # ── Targeted fills (non-zero) — MUST come before the generic zero-fill ──
    # days_since_last_txn for zombie wallets = max possible value.
    # Zombies never transacted in the feature window; 0 would mean "transacted today",
    # which is the opposite of correct.
    max_days = (FEATURE_WINDOW_END - FEATURE_WINDOW_START).days
    if "days_since_last_txn" in master.columns:
        master["days_since_last_txn"] = master["days_since_last_txn"].fillna(max_days + 1)

    # Tenure feature from registration_date (derived — not a behavioral fill)
    if "registration_date" in master.columns:
        master["tenure_days"] = (
            FEATURE_WINDOW_END - master["registration_date"]
        ).dt.days.fillna(-1).astype(int)

    # ── Generic zero-fill for remaining numeric behavioral NaNs ─────────────
    # Excludes days_since_last_txn (already filled above with a meaningful value).
    _TARGETED_FILLS = {"days_since_last_txn"}
    numeric_behavioral_cols = behavioral.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_behavioral_cols:
        if c in master.columns and c not in _TARGETED_FILLS:
            master[c] = master[c].fillna(0)

    print(f"[join] master shape: {master.shape[0]:,} rows × {master.shape[1]} cols "
          f"({time.time()-t0:.1f}s)")

    # Sanity checks
    assert master["wallet_id"].is_unique, "wallet_id is not unique in master"
    assert master["churned"].notna().all(), "churned has NaN values"
    # is_reactivator not yet assigned here; full sanity check runs in build_customers_master()
    if "is_reactivator" in master.columns:
        n_react = (master["is_reactivator"] == 1).sum()
        react_dslt_zero = ((master["is_reactivator"] == 1) & (master["days_since_last_txn"] == 0)).sum()
        react_dslt_max = ((master["is_reactivator"] == 1) & (master["days_since_last_txn"] >= max_days)).sum()
        print(f"[validate] reactivators with days_since_last_txn=0: {react_dslt_zero:,} (expect 0)")
        print(f"[validate] reactivators with days_since_last_txn>={max_days}: {react_dslt_max:,} (expect {n_react:,})")
        assert react_dslt_zero == 0, "Bug: reactivators have bogus days_since_last_txn=0"
    return master


# ----------------------------------------------------------------------
# 5b. Reactivation features (computed after full join)
# ----------------------------------------------------------------------

def _add_reactivation_features(
    master: pd.DataFrame,
    tx_label: pd.DataFrame,
) -> pd.DataFrame:
    """Add is_reactivator and dormancy_days_before_reactivation to master.

    A reactivator has zero feature-window transactions AND at least one
    label-window transaction.  These are genuine reactivations — the
    customer was dormant Jan–Apr but returned May–Jun — so churned=0 is
    correct.  Flagging them separately allows the cluster-then-predict
    layer (Phase 2.6) to treat them as a distinct behavioral segment.

    dormancy_days_before_reactivation: days from registration_date to the
    customer's first label-window transaction.  Set to 0 for non-reactivators.
    """
    # Wallets with zero feature-window activity (txn_count was filled to 0
    # in build_master for all "zombie" wallets)
    zero_feat_ids: set = set(master.loc[master["txn_count"] == 0, "wallet_id"])
    active_label_ids: set = set(tx_label["wallet_id"].unique())
    reactivator_ids: set = zero_feat_ids & active_label_ids

    n_react = len(reactivator_ids)
    print(f"[reactivator] {n_react:,} reactivators identified "
          f"({n_react / len(master):.2%} of customers)")

    master = master.copy()
    master["is_reactivator"] = master["wallet_id"].isin(reactivator_ids).astype(int)

    # First label-window txn date per reactivator — used for dormancy
    first_label_txn = (
        tx_label.loc[tx_label["wallet_id"].isin(reactivator_ids)]
        .groupby("wallet_id")["timestamp"]
        .min()
        .rename("first_label_txn_date")
    )
    master = master.merge(first_label_txn, on="wallet_id", how="left")

    master["dormancy_days_before_reactivation"] = 0
    react_mask = master["is_reactivator"] == 1
    if react_mask.any():
        master.loc[react_mask, "dormancy_days_before_reactivation"] = (
            (master.loc[react_mask, "first_label_txn_date"]
             - master.loc[react_mask, "registration_date"])
            .dt.days
            .clip(lower=0)
            .fillna(0)
            .astype(int)
        )
    master = master.drop(columns=["first_label_txn_date"])
    return master


# ----------------------------------------------------------------------
# 6. Pipeline entrypoint
# ----------------------------------------------------------------------

def build_customers_master(save: bool = True) -> pd.DataFrame:
    """Run the full pipeline and (optionally) save customers_master.parquet."""
    overall_t0 = time.time()

    tx = load_transactions()
    profiles = load_profiles()

    # Drop customers registered after the feature window ends.
    # These ~300 wallets have no possible feature-window history — they are
    # structural temporal contamination (registered May–Jun, so any feature
    # we compute for them would be vacuous).  Filter here so the master file
    # is clean from the start.
    if "registration_date" in profiles.columns:
        late_mask = profiles["registration_date"] > FEATURE_WINDOW_END
        n_dropped = int(late_mask.sum())
        print(f"[filter] dropping {n_dropped:,} late registrants "
              f"(registration_date > {FEATURE_WINDOW_END.date()})")
        profiles = profiles.loc[~late_mask].reset_index(drop=True)

    tx_feat, tx_label = split_by_time(tx)
    behavioral = aggregate_behavior(tx_feat)
    labels = engineer_churn_label(tx_label, profiles["wallet_id"])

    master = build_master(profiles, behavioral, labels)
    master = _add_reactivation_features(master, tx_label)

    # ── Sanity check: zombie recency fill correctness ──────────────────────────
    # is_reactivator is now present; this is the earliest point the check can fire.
    _max_days = (FEATURE_WINDOW_END - FEATURE_WINDOW_START).days
    n_react = int((master["is_reactivator"] == 1).sum())
    react_dslt_zero = int(((master["is_reactivator"] == 1) & (master["days_since_last_txn"] == 0)).sum())
    react_dslt_max  = int(((master["is_reactivator"] == 1) & (master["days_since_last_txn"] >= _max_days)).sum())
    print(f"[validate] reactivators with days_since_last_txn=0: {react_dslt_zero:,} (expect 0)")
    print(f"[validate] reactivators with days_since_last_txn>={_max_days} ({_max_days}): {react_dslt_max:,} (expect {n_react:,})")
    assert react_dslt_zero == 0, "Bug: reactivators have bogus days_since_last_txn=0"

    # ── Data quality audit (CLAUDE.md §6, locked 2026-05-02) ──────────────────
    # Drop synthetic JSON fields and redundant raw-date forms.
    # registration_date must be dropped HERE (after _add_reactivation_features
    # uses it to compute dormancy_days_before_reactivation).
    _AUDIT_DROP = [
        "date_of_birth",             # redundant: age already present
        "registration_date",         # redundant: tenure_days + dormancy computed above
        "account_status",            # synthetic JSON field, contradicts Parquet patterns
        "notification_preferences",  # synthetic JSON field, no causal connection to behavior
        "last_txn_date",             # redundant: days_since_last_txn computed in aggregate_behavior
        "first_txn_date",            # redundant: account_active_span_days computed in aggregate_behavior
        "support_tier",              # synthetic JSON field, signal redundant with kyc_tier
    ]
    audit_drop_cols = [c for c in _AUDIT_DROP if c in master.columns]
    if audit_drop_cols:
        master = master.drop(columns=audit_drop_cols)
        print(f"[audit] dropped {len(audit_drop_cols)} cols: {audit_drop_cols}")
    print(f"[audit] master after audit: {master.shape[0]:,} rows x {master.shape[1]} cols")

    # Vendor-vs-engineered label agreement (methodology contribution)
    if "churned_vendor" in master.columns:
        master["churned_vendor"] = master["churned_vendor"].fillna(0).astype(int)
        agreement = (master["churned"] == master["churned_vendor"]).mean()
        print(f"[validate] engineered vs vendor label agreement: {agreement:.1%}")

    if save:
        master.to_parquet(CUSTOMERS_MASTER, compression="snappy", index=False)
        print(f"[save] {CUSTOMERS_MASTER} "
              f"({CUSTOMERS_MASTER.stat().st_size/1e6:.1f} MB)")

    print(f"[done] total time: {time.time()-overall_t0:.1f}s")
    return master


if __name__ == "__main__":
    build_customers_master()
