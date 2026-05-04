# RetainIQ — Churn Prediction for Digital Wallets

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eliekh2/315-REPO/blob/main/notebooks/00_master_submission.ipynb)

ML pipeline for predicting customer churn in mobile money platforms.
Built on 4M Nigerian mobile money transactions and 375K customer profiles.
AUB MSBA 315, Spring 2026 — solo project by Elie Khayrallah.

---

## Quick start

**Option 1 — Google Colab (recommended):** Click the badge above. Runtime → Restart and Run All. Done in ~15 minutes — no local setup.

**Option 2 — Local:**

```bash
git clone https://github.com/Eliekh2/315-REPO.git
cd 315-REPO
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
jupyter notebook notebooks/00_master_submission.ipynb
```

---

## Results

| Model | AUC | PR-AUC | Precision@10% |
|---|---|---|---|
| Logistic_Optimized | 0.7078 | 0.2570 | 0.2946 |
| Logistic_Vanilla | 0.7078 | 0.2570 | 0.2942 |
| RandomForest_Tuned | 0.7571 | 0.4223 | 0.3885 |
| XGBoost_Vanilla | 0.7591 | 0.4255 | 0.3874 |
| XGBoost_Tuned | 0.7590 | 0.4258 | 0.3880 |
| ClusterPredict_XGB | 0.7585 | 0.4243 | 0.3888 |

**Key findings:**
- Tree models outperform LR by +0.051 AUC, confirming non-linear churn dynamics
- XGBoost (Optuna-tuned) is the primary model: AUC 0.759, 38.9% precision in top decile (3.8× lift over random)
- K-means clustering recovers a 100%-churn reactivator segment without labels → automatic retention trigger rule
- Ceiling at ~0.76 AUC across all tree models: the data's predictive signal is approximately fully captured

---

## Documents

- **📖 [Project Story](docs/PROJECT_STORY.md)** — start here if you're new to the project
- **🐛 [Issues & Fixes](docs/ISSUES_AND_FIXES.md)** — every non-trivial problem encountered and resolved
- **🗂 [Project Plan](CLAUDE.md)** — full context, locked decisions, status tracker

---

## Project structure

```
retainiq/
├── notebooks/
│   ├── 00_master_submission.ipynb   ← submitted file (Colab-ready)
│   ├── 01_eda.ipynb
│   ├── 02_logistic_baseline.ipynb
│   ├── 03_random_forest.ipynb
│   ├── 04_xgboost.ipynb
│   └── 06_cluster_then_predict.ipynb
├── src/                             ← Python modules (config, evaluate, etc.)
├── data/processed/                  ← train.parquet, test.parquet, master
├── outputs/
│   ├── models/                      ← .pkl files for all trained models
│   ├── tables/benchmark.csv         ← canonical results table
│   └── figures/                     ← all plots (dpi=150)
└── docs/
    ├── research_paper.md
    ├── PROJECT_STORY.md
    ├── ISSUES_AND_FIXES.md
    └── ai_usage_log.md
```

---

## Authors

Elie Khayrallah, MSBA 315, American University of Beirut, Spring 2026.
Contact: elieekhairallah@gmail.com

## License

MIT for code. Dataset: Nigerian Mobile Money Dataset (synthetic, public domain).
