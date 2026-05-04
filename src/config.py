"""
Project-wide configuration: paths, seeds, time windows, constants.
Import from anywhere via: from src.config import *
All locked decisions from CLAUDE.md §6 live here.
"""
from pathlib import Path
import pandas as pd

# === Paths ===
# Repo root = parent folder of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source data sits OUTSIDE the repo, in the parent "315 - retain IQ" folder
DATA_RAW = PROJECT_ROOT.parent / "Datasets selected"
PARQUET_PATH = DATA_RAW / "Nigerian datasets PARQUET" / "nigerian_mobile_money_full.parquet"
JSON_PATH = DATA_RAW / "Customer Profiles JSON" / "customer_profiles.json"

# Processed data
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CUSTOMERS_MASTER = DATA_PROCESSED / "customers_master.parquet"
TRAIN_PATH = DATA_PROCESSED / "train.parquet"
TEST_PATH = DATA_PROCESSED / "test.parquet"

# Outputs
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
TABLES = OUTPUTS / "tables"
BENCHMARK_CSV = TABLES / "benchmark.csv"

# Ensure directories exist
for p in [DATA_INTERIM, DATA_PROCESSED, FIGURES, MODELS, TABLES]:
    p.mkdir(parents=True, exist_ok=True)

# === Reproducibility ===
RANDOM_SEED = 42

# === Train/test split ===
TEST_SIZE = 0.20
N_SPLITS_CV = 5

# === Target column ===
TARGET = "churned"  # customer-level, engineered in data_loader

# === Locked time windows (CLAUDE.md §6) ===
# 6 months total: Jan 2024 – Jun 2024
# Feature window: first 4 months — used to compute ALL behavioral features
# Label window:   last 2 months — used ONLY for the churn label
# These windows must NEVER overlap. No feature ever touches label-window data.
FEATURE_WINDOW_START = pd.Timestamp("2024-01-01")
FEATURE_WINDOW_END = pd.Timestamp("2024-04-30")
LABEL_WINDOW_START = pd.Timestamp("2024-05-01")
LABEL_WINDOW_END = pd.Timestamp("2024-06-30")

# === Customer-level churn definition (CLAUDE.md §6) ===
# A customer is "churned" (=1) if they have ZERO transactions in the label window.
# Otherwise they are "retained" (=0).
# Engineered from raw activity, independent of the vendor's churn_30d field.
# The vendor field is kept as `churned_vendor` for sanity-check reporting only.
