# Phase 3.2 — Colab-Ready Master Notebook Packaging

> **For Claude Code:** This document is consumed at Phase 3.2 (per `CLAUDE.md` §5). All Phase 1 and Phase 2 work must be complete before starting. Read this in full, then read `CLAUDE.md` to refresh context, then execute the tasks below in order.

> **For Elie:** Save this file as `docs/PHASE_3_2_COLAB_PACKAGING.md` in your repo. When you reach Phase 3.2, open Claude Code and paste: *"Read `docs/PHASE_3_2_COLAB_PACKAGING.md` and `CLAUDE.md`, then execute the tasks in PHASE_3_2_COLAB_PACKAGING.md in order. The GitHub repo URL and Google Drive folder are already locked in §2 of the doc — do not prompt me for them. This phase produces three deliverables: (1) the master Colab-ready notebook that consolidates all phase notebooks into one, with code+data fetched from GitHub at runtime; (2) `docs/ISSUES_AND_FIXES.md`; (3) `docs/PROJECT_STORY.md`."*

---

## 0. Goal

Phase 3.2 produces **three deliverables** that together complete the project:

### Deliverable 1 — Master Colab-ready notebook (`notebooks/00_master_submission.ipynb`)

**One** clean, professional notebook that:

- Anyone (specifically Prof. Wael Khreich) can open in Google Colab
- Click "Run All" with **zero local setup**
- Have it run end-to-end without errors in under 30 minutes
- Reproduce every result, table, and figure cited in the research paper

The master notebook **consolidates** the per-phase development notebooks (`01_eda`, `02_logistic_baseline`, `03_random_forest`, `04_xgboost`, `06_cluster_then_predict`, `08_shap_analysis`, etc.) into one coherent narrative — see **Section 2.5** for the consolidation procedure. The dev notebooks remain in the repo as appendix material; the master is the single submitted file.

The notebook fetches all source code (`src/`), processed data (`data/processed/*.parquet`), and serialized models (`outputs/models/*.pkl`) from the GitHub repo `https://github.com/Eliekh2/315-REPO` at runtime — no upload, no auth, no setup required from the professor.

This is the implementation deliverable that satisfies the project description's *"A Jupyter Notebook with a clear structure and comments describing your code"* requirement and the rubric's 25% Implementation grade (code clarity, structure, comments, replicability).

### Deliverable 2 — Issues & Fixes log (`docs/ISSUES_AND_FIXES.md`)

A documented record of every non-trivial problem encountered during development and how it was resolved. Feeds the report's Methodology and Discussion sections; demonstrates rigor and earns Optimization & Novelty points (worth 20 in the rubric).

Full spec in **Section 9**.

### Deliverable 3 — Storytelling document (`docs/PROJECT_STORY.md`)

A 3,000–5,000 word prose narrative walking a non-technical reader through the entire project from problem to solution. Doubles as a portfolio piece, an onboarding doc for future contributors, and a layperson-accessible summary for stakeholders (Whish Money, OMT Pay, etc.).

Full spec in **Section 10**.

---

## 1. Pre-flight checklist (verify before starting)

Confirm all of the following exist and are complete. If any is missing, STOP and tell Elie:

- [ ] `data/processed/customers_master.parquet` exists (~28 MB, 375K rows, ~45 cols)
- [ ] `data/processed/train.parquet` and `data/processed/test.parquet` exist (frozen 80/20 stratified split)
- [ ] `outputs/tables/benchmark.csv` has at least 6 model rows (one per Phase 2 track)
- [ ] `outputs/models/` contains `.pkl` files for each tuned model
- [ ] `outputs/figures/` has all EDA, evaluation, and SHAP plots saved at dpi=150
- [ ] `notebooks/01_eda.ipynb` through `notebooks/08_shap_analysis.ipynb` all run top-to-bottom without errors
- [ ] `src/config.py`, `src/data_loader.py`, `src/evaluate.py`, `src/features.py` are stable
- [ ] All Phase 1 and Phase 2 status items in `CLAUDE.md` §13 are checked

If any item fails: stop, document what's missing, and surface it to Elie before proceeding.

---

## 2. Locked external resources (URLs already confirmed by Elie)

The hosting decisions are LOCKED — do not prompt Elie for them, do not invent placeholders, do not propose alternatives.

### GitHub repository (source code + processed data + serialized models)

```
GITHUB_URL = "https://github.com/Eliekh2/315-REPO"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Eliekh2/315-REPO/main"
```

This is the canonical source for everything reproducible:
- All source code (`src/`)
- All notebooks (`notebooks/`)
- The processed datasets — `data/processed/customers_master.parquet`, `train.parquet`, `test.parquet`
- All serialized models (`outputs/models/*.pkl`)
- The master notebook (`notebooks/00_master_submission.ipynb`)

### Hosting strategy (locked: GitHub primary, Google Drive backup only)

Verify each parquet file's size BEFORE pushing to GitHub:

```python
import os
from src.config import CUSTOMERS_MASTER, TRAIN_PATH, TEST_PATH
for p in [CUSTOMERS_MASTER, TRAIN_PATH, TEST_PATH]:
    print(f"{p.name}: {os.path.getsize(p)/1e6:.1f} MB")
```

- **All three files under 50 MB each:** push directly to GitHub. Done.
- **Any file 50–100 MB:** still push to GitHub, but add a `.gitattributes` entry to use Git LFS for that file. Document in README that `git-lfs` is needed for local clones.
- **Any file over 100 MB:** that file goes to the Google Drive fallback below; the others stay on GitHub.

The model `.pkl` files: same rule. Most should be small. The Random Forest pkl was 26 MB (well under). XGBoost is typically smaller. CatBoost can be larger — check.

### Google Drive backup (fallback ONLY if a file exceeds GitHub limits)

```
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1wd9_B0ZmopkYxfzxTsDmEHD9iI7vH8YK?usp=sharing"
```

Only use Google Drive for files that genuinely cannot fit on GitHub. If used, the folder must be set to "Anyone with the link → Viewer". Each file's individual share link must be obtained (not the folder link) and downloaded via `gdown` in the master notebook.

### Sensitive data check

Confirm before pushing anything public: customer wallet IDs are synthetic identifiers from the public Nigerian mobile money dataset; profile data is synthetic; no real PII. Safe to publish.

If anything in the repo looks like real PII, stop and surface to Elie before pushing.

---

## 2.5 Notebook Consolidation Procedure

The dev workflow produced multiple per-phase notebooks (`01_eda.ipynb`, `02_logistic_baseline.ipynb`, `03_random_forest.ipynb`, `04_xgboost.ipynb`, `06_cluster_then_predict.ipynb`, `08_shap_analysis.ipynb`, etc.). These remain in the repo as development artifacts. The master notebook (`notebooks/00_master_submission.ipynb`) is a NEW file that **consolidates the meaningful cells from each into a single coherent narrative**.

### Consolidation principles

**Not a literal merge.** The master is not "concat all dev notebooks." It is a curated, narrative-driven rewrite where each section pulls only the meaningful cells from its corresponding dev notebook. Throw away exploratory dead ends, debug prints, sanity checks that don't affect the result, and intermediate diagnostic cells that were useful during development but have served their purpose.

**Single coherent voice.** Every markdown cell in the master is rewritten from scratch in the human-researcher voice (no "delve," "moreover," "furthermore," "comprehensive," "leverage," "underscore"). Even where the dev notebook's markdown is fine, prefer rewriting from a clean perspective so the document reads as one piece, not stitched fragments.

**Idempotent execution.** The master must run end-to-end without depending on any state from prior cells in the dev notebooks. Variables created in one master section must not silently rely on variables from another. Each section starts cleanly.

**Skip-training default.** All model sections default to loading from `outputs/models/*.pkl` (downloaded via Cell 7). A `SKIP_TRAINING` flag at the top of each model section lets the professor toggle retraining if they want — but the default path is load-and-evaluate, which keeps total runtime under 15 minutes.

### Per-section consolidation map

For each section in §3 below, the master notebook pulls from these sources:

| Master section | Source dev notebook(s) | What to pull | What to drop |
|---|---|---|---|
| C1 — Data Acquisition | `01_eda.ipynb` (header) + `data_loader.py` | Schema overview, source descriptions | Loader execution itself (loader is documented, not run) |
| C2 — Preprocessing & Feature Engineering | `01_eda.ipynb` + `data_loader.py` docstring | Time-window logic, churn label definition, audit decisions | Detailed audit deliberations |
| C3 — EDA | `01_eda.ipynb` | 4–6 most important figures (class balance, churn by `account_status` if kept, top correlations, reactivator analysis) | All other EDA |
| C4 — Train/Test Split | `01b_train_test_split.ipynb` (or equivalent) | Stratified split logic + verification | Diagnostic prints |
| C5 — Logistic Regression | `02_logistic_baseline.ipynb` | Final tuned model load + evaluation + coefficient analysis | Vanilla LR (mention as comparison only), all the recency/multicollinearity diagnostic cells |
| C6 — Random Forest | `03_random_forest.ipynb` | Tuned RF load + evaluation + permutation importance | Vanilla RF if not in benchmark, OOB analysis details |
| C7 — XGBoost (primary) | `04_xgboost.ipynb` | Tuned XGB load + evaluation + 3-method feature importance | Vanilla XGB load (mention only), Optuna trial logs |
| C8 — Class Imbalance Study | (skip if Phase 2.5 not run) | — | — |
| C9 — Cluster-then-Predict (novelty) | `06_cluster_then_predict.ipynb` | K selection, cluster profiles, per-cluster + aggregated AUC, feature heterogeneity heatmap | DBSCAN sensitivity check unless it produced a finding |
| C10 — Benchmark Synthesis | `outputs/tables/benchmark.csv` | Final results table sorted by primary metric | — |
| C11 — SHAP Analysis | `08_shap_analysis.ipynb` | Global summary + 3 waterfall plots | Detailed SHAP value tables |
| C12 — Conclusions & Recommendations | (new content for master) | Mirrors paper §7 | — |

### Consolidation workflow

1. Generate `notebooks/00_master_submission.ipynb` from scratch using `nbformat`, not by copying-and-modifying existing notebooks.
2. For each master section, programmatically read the relevant dev notebook(s), identify the cells to pull, and rewrite the markdown surrounding them.
3. After every section is written, restart kernel and run all in a clean Jupyter environment (locally first). Fix anything that errors.
4. THEN test in clean Colab incognito (Section 6).
5. Commit master notebook to GitHub.
6. Generate Colab badge.

### Files preserved vs hidden

The dev notebooks remain in `notebooks/` for inspection. The master notebook is the only one referenced from the README's "Open in Colab" badge. The professor opens the master; the dev notebooks are appendix material.

---

## 3. The master notebook structure

Build `notebooks/00_master_submission.ipynb` with **exactly** the following cell sequence. Do not add or remove sections without flagging Elie first.

### Section A — Front matter (markdown cells)

**Cell 1 — Title block:**
- Project title: "RetainIQ: Customer Churn Prediction for Digital Wallets"
- Course: MSBA 315 — Spring 2026, AUB
- Author: Elie [last name]
- Submission date
- A "How to run this notebook" subsection with three options:
  - **Option 1 — Google Colab (recommended):** "Open in Colab" badge link generated from the GitHub URL using the format `https://colab.research.google.com/github/{user}/{repo}/blob/main/notebooks/00_master_submission.ipynb`
  - **Option 2 — Locally:** clone the repo, install `requirements.txt`, open in Jupyter
  - **Option 3 — Inspection only:** view rendered notebook on GitHub without running it
- Estimated runtime: state the actual runtime measured during testing (likely 10–25 min)
- RAM requirement: ~4 GB (Colab free tier provides 12 GB → fine)

**Cell 2 — Abstract / Executive Summary:**
- 200–300 words mirroring the report abstract
- States the problem, dataset, methodology, and headline result (best AUC, recall@10%, etc.)

### Section B — Environment setup (code cells)

**Cell 3 — Install dependencies:**
```python
!pip install -q pandas numpy pyarrow scikit-learn xgboost lightgbm catboost \
    imbalanced-learn shap optuna matplotlib seaborn
```
Use `-q` for quiet mode. Pin versions if any compatibility issue surfaced during testing. (`gdown` is installed conditionally in Cell 6 only if a Google Drive fallback is needed.)

**Cell 4 — Detect environment (Colab vs local):**
```python
import os, sys

IN_COLAB = 'google.colab' in sys.modules
print(f"Running in: {'Google Colab' if IN_COLAB else 'Local Jupyter'}")
```

**Cell 5 — Get source code (clone the GitHub repo on Colab):**
```python
GITHUB_URL = "https://github.com/Eliekh2/315-REPO.git"
REPO_DIR = "315-REPO"

if IN_COLAB and not os.path.exists(REPO_DIR):
    !git clone -q {GITHUB_URL}
    %cd {REPO_DIR}
elif not IN_COLAB:
    # Already in local repo, no-op
    pass

# Make src/ importable
sys.path.insert(0, os.getcwd())
print(f"Working dir: {os.getcwd()}")
print(f"src/ accessible: {os.path.exists('src')}")
```

**Cell 6 — Get processed data (from GitHub by default, Google Drive fallback):**
```python
# DATA HOSTING POLICY (locked):
# - Primary source: GitHub repo (already cloned in Cell 5)
# - The data files live at data/processed/ in the cloned repo
# - If git clone succeeded, the files are already on disk — nothing to download
#
# If a file was too large for GitHub and lives on Google Drive instead, the
# fallback block below will detect missing files and pull them via gdown.

import os
from pathlib import Path

DATA_DIR = Path("data/processed")
REQUIRED_FILES = ["customers_master.parquet", "train.parquet", "test.parquet"]

# Per-file Google Drive direct-download IDs (only populated if a file is too
# large for GitHub — leave empty strings if all files are on GitHub)
GDRIVE_FILE_IDS = {
    "customers_master.parquet": "",  # populate ONLY if GitHub didn't host it
    "train.parquet":             "",
    "test.parquet":              "",
}

missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
if missing:
    print(f"Missing files (need Google Drive fallback): {missing}")
    !pip install -q gdown
    import gdown
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in missing:
        fid = GDRIVE_FILE_IDS.get(fname, "")
        if not fid:
            raise FileNotFoundError(
                f"{fname} is missing from the GitHub clone and no Google Drive "
                f"fallback ID is configured. Either push the file to GitHub or "
                f"add its file_id to GDRIVE_FILE_IDS in this cell."
            )
        gdown.download(f"https://drive.google.com/uc?id={fid}",
                       str(DATA_DIR / fname), quiet=False)

# Verify
import pandas as pd
for f in REQUIRED_FILES:
    df = pd.read_parquet(DATA_DIR / f)
    print(f"  ✓ {f}: {df.shape[0]:,} rows × {df.shape[1]} cols")
```

**Cell 7 — Get pretrained models (from GitHub):**
```python
# Models live in outputs/models/ within the cloned repo. If git clone succeeded,
# they're already on disk. Set SKIP_TRAINING=False to retrain everything from
# scratch (adds 30–60 min runtime — generally not what the professor wants).

SKIP_TRAINING = True

import os, joblib
from pathlib import Path

MODELS_DIR = Path("outputs/models")
EXPECTED_MODELS = [
    "logistic.pkl",
    "random_forest.pkl",
    "xgboost.pkl",
    # Add any cluster_predict pkls if they were saved per-cluster, e.g.:
    # "xgb_cluster_0.pkl", "xgb_cluster_1.pkl", ...
]

if SKIP_TRAINING:
    found, missing = [], []
    for name in EXPECTED_MODELS:
        if (MODELS_DIR / name).exists():
            found.append(name)
        else:
            missing.append(name)
    print(f"Found models: {found}")
    if missing:
        print(f"Missing models (will need retraining or fallback): {missing}")
else:
    print("SKIP_TRAINING=False — models will be retrained in their respective sections")
```

**Cell 8 — Import project modules and configuration:**
```python
from src.config import (
    RANDOM_SEED, TARGET, FIGURES, MODELS, TABLES, BENCHMARK_CSV,
    TRAIN_PATH, TEST_PATH, CUSTOMERS_MASTER,
)
from src.evaluate import evaluate, append_to_benchmark, print_metrics
from src.data_loader import build_customers_master  # available but not run by default

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

print("Setup complete. Random seed:", RANDOM_SEED)
```

### Section C — Methodology walkthrough (mirrors the report)

Each subsection corresponds to a section in the research paper. Each starts with a markdown cell summarizing what's happening and why, then has the runnable code, then a markdown cell interpreting the output.

**Section C1 — Data Acquisition (mirrors report §5.1)**
- Markdown: describe the two source files (Parquet + JSON), schema, time range, the MSBA 305 pipeline that produced them
- Code: load `customers_master.parquet`, print shape, print head(5), print dtypes
- Markdown: explain why we work at customer-grain (not transaction-grain), and why JSON profiles are joined onto aggregated transactions (not the reverse). Reference the locked decisions in CLAUDE.md §6.

**Section C2 — Preprocessing & Feature Engineering (mirrors report §5.2)**
- Markdown: explain the time-window split (Jan–Apr features, May–Jun labels), the leakage-prevention rationale, and the customer-level churn label definition
- Code: optionally re-run `build_customers_master()` (gated behind a `REBUILD_MASTER = False` flag — default False because data is downloaded pre-built)
- Markdown: summarize the engineered features (~28 behavioral + 14 profile + reactivation flags). Reference the reactivator finding from EDA.

**Section C3 — Exploratory Data Analysis (mirrors report §5.3, abbreviated)**
- Pull the 4–6 most important figures from `notebooks/01_eda.ipynb` only. Do not redo all EDA — this is a master narrative, not a dump.
- Required: class balance plot, churn rate by `account_status`, top-10 feature correlations, reactivator subsection
- Each figure saved during execution to `outputs/figures/master/` so the master notebook is self-contained

**Section C4 — Train/Test Split (mirrors report §5.4)**
- Markdown: stratified 80/20, frozen, seed 42
- Code: load `train.parquet` and `test.parquet`, print shapes and class balance
- Confirm the test set has not been touched

**Section C5 — Modeling Track 1: Logistic Regression Baseline**
- Markdown: rationale (interpretable baseline, RFE feature selection, calibration)
- Code: load pretrained model OR retrain (gated by `SKIP_TRAINING`); evaluate on test; append to benchmark
- Markdown: result interpretation (1–2 sentences)

**Section C6 — Modeling Track 2: Random Forest**
**Section C7 — Modeling Track 3: XGBoost (primary model)**
**Section C8 — Modeling Track 4: LightGBM + CatBoost**
**Section C9 — Modeling Track 5: Class Imbalance Study**
**Section C10 — Modeling Track 6: Cluster-then-Predict (novelty layer)**

For each: same pattern as C5 — markdown rationale, code (load pretrained or retrain), result interpretation. Keep code blocks tight; if a section needs more than ~80 lines, refactor into a helper function in `src/`.

**Section C11 — Benchmark Synthesis**
- Code: load `outputs/tables/benchmark.csv`, format as a pandas DataFrame, sort by chosen primary metric (likely PR-AUC for imbalanced problem)
- Markdown table summarizing the comparison
- Bar chart of AUC across all models

**Section C12 — SHAP Analysis on Winning Model**
- Pull the 3–4 key SHAP figures from `notebooks/08_shap_analysis.ipynb`
- Global feature importance plot
- Summary plot (beeswarm)
- 3 force/waterfall plots: high-risk, mid-risk, low-risk customer
- Markdown interpretation tying back to the literature review's expected feature ranking

**Section C13 — Business Recommendations & Conclusions**
- Markdown only
- Mirrors report §7
- 3–5 concrete recommendations for a deploying customer (e.g., Whish Money)
- Limitations
- Future work

### Section D — Appendix (markdown cells)

**Cell — Reproducibility statement:**
- Random seed
- Library versions (auto-detected via `!pip freeze` or `pkg_resources`)
- Hardware tested on (Colab T4 GPU, Colab CPU, local Windows)
- Total notebook runtime measured

**Cell — Citation:**
- Suggested citation for the project
- License (e.g., MIT for code, dataset citation for data)

**Cell — Acknowledgments:**
- Prof. Wael Khreich
- AUB MSBA program
- Anthropic (Claude was used as an AI development partner per the AI usage log in `docs/ai_usage_log.md`)

---

## 4. GitHub commit procedure (final push)

Once the master notebook is built and tested locally, push everything to `https://github.com/Eliekh2/315-REPO`:

```bash
# From the repo root
git status                                      # review what's changed
git add notebooks/00_master_submission.ipynb
git add data/processed/*.parquet                # only if the .gitignore override is in place (see §7)
git add outputs/models/*.pkl
git add outputs/tables/benchmark.csv
git add outputs/figures/                        # at least the key figures used in the master notebook
git add docs/                                   # ISSUES_AND_FIXES, PROJECT_STORY, etc.
git add CLAUDE.md README.md requirements.txt .gitignore
git commit -m "Phase 3.2: master notebook + processed data + serialized models for Colab reproduction"
git push origin main
```

Verify the push succeeded by visiting `https://github.com/Eliekh2/315-REPO` and confirming:
- `notebooks/00_master_submission.ipynb` is visible
- `data/processed/customers_master.parquet`, `train.parquet`, `test.parquet` are visible
- `outputs/models/*.pkl` are visible
- README has the Colab badge

If any file failed to push because of size:
1. Identify which file via `git push` error output
2. Move it to the Google Drive fallback folder (`https://drive.google.com/drive/folders/1wd9_B0ZmopkYxfzxTsDmEHD9iI7vH8YK`)
3. Get its individual share link → file_id
4. Update Cell 6's `GDRIVE_FILE_IDS` dict in the master notebook
5. Add the file to `.gitignore`
6. Re-attempt push without that file

---

## 5. Generate the "Open in Colab" badge

The badge URL is fixed (GitHub repo is locked):

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eliekh2/315-REPO/blob/main/notebooks/00_master_submission.ipynb)
```

Embed this badge in:
- The top markdown cell of the master notebook itself (Cell 1)
- The repo's `README.md`

Verify by clicking the badge in a browser after pushing to GitHub. It should open Colab with the master notebook loaded. If it 404s, the path or repo name is wrong — fix before declaring done.

---

## 6. Mandatory testing protocol

Before declaring Phase 3.2 complete, you MUST verify the notebook runs cleanly in a fresh Colab session. This is the most important step in the entire phase.

**Procedure:**

1. After all code is committed to GitHub (`https://github.com/Eliekh2/315-REPO`) and any Google-Drive fallback files are uploaded, do this:
   - Open https://colab.research.google.com in **incognito mode** (no logged-in Colab cache)
   - File → Open notebook → GitHub tab → paste `Eliekh2/315-REPO` → select `notebooks/00_master_submission.ipynb`
   - Runtime → Restart and Run All
   - Walk away for 30 minutes
   - Return and check: did every cell complete without errors? Did all figures render? Did the benchmark table appear?

2. If any cell errors, debug and re-test from a clean session. **Do not skip the clean re-test** — partial cache from previous runs hides bugs.

3. Document the verified Colab runtime in the notebook's front matter (Cell 1).

4. If successful, mark Phase 3.2 complete in CLAUDE.md §13.

---

## 7. Files to commit to GitHub

When pushing the repo for Colab access, ensure these are committed:

```
315-REPO/                                  (root of the GitHub repo)
├── CLAUDE.md
├── README.md (with Colab badge)
├── requirements.txt
├── .gitignore
├── src/                                ← all .py files
├── notebooks/
│   ├── 00_master_submission.ipynb     ← the submitted file
│   ├── 01_eda.ipynb                   ← reference (dev artifact)
│   ├── 02_logistic_baseline.ipynb     ← reference
│   ├── ... (all dev notebooks)        ← reference, not run by professor
│   └── 08_shap_analysis.ipynb
├── data/
│   └── processed/
│       ├── customers_master.parquet   ← commit (under 50 MB)
│       ├── train.parquet              ← commit (under 50 MB)
│       └── test.parquet               ← commit (under 50 MB)
├── outputs/
│   ├── tables/benchmark.csv           ← important, commit this
│   ├── models/*.pkl                   ← commit (verify each under 50 MB)
│   └── figures/                       ← commit key figures only
└── docs/
    ├── research_paper.md
    ├── presentation.pptx
    ├── ai_usage_log.md
    ├── ISSUES_AND_FIXES.md             ← new (Section 9)
    ├── PROJECT_STORY.md                ← new (Section 10)
    └── PHASE_3_2_COLAB_PACKAGING.md    ← this file
```

Do NOT commit:
- `Datasets selected/` (raw source data, not in the repo anyway per `.gitignore`) — note that the Phase 3.2 commit will need to OVERRIDE the existing `.gitignore` rule for `data/processed/*.parquet`. Update `.gitignore` to allow the three processed files specifically:
  ```
  data/processed/*
  !data/processed/customers_master.parquet
  !data/processed/train.parquet
  !data/processed/test.parquet
  ```
- `.venv/`
- Jupyter checkpoints
- Intermediate files in `data/interim/`

### File size verification before push

Run this BEFORE `git push` to confirm everything fits:

```bash
python -c "
from pathlib import Path
p = Path('data/processed')
for f in sorted(p.glob('*.parquet')):
    mb = f.stat().st_size / 1e6
    flag = '✗ TOO LARGE' if mb > 50 else '✓'
    print(f'{flag}  {f.name}: {mb:.1f} MB')
p = Path('outputs/models')
for f in sorted(p.glob('*.pkl')):
    mb = f.stat().st_size / 1e6
    flag = '✗ TOO LARGE' if mb > 50 else '✓'
    print(f'{flag}  {f.name}: {mb:.1f} MB')
"
```

If any file is over 50 MB:
1. Move that specific file to the Google Drive fallback folder
2. Get its individual share link → file_id
3. Populate the corresponding entry in Cell 6's `GDRIVE_FILE_IDS` dict
4. Add the file to `.gitignore` so git doesn't try to push it

Do not commit a file over 50 MB to GitHub without setting up Git LFS first; doing so will fail the push.

---

## 8. README.md update for the GitHub repo

After Phase 3.2 completes, the repo's `README.md` should be rewritten to be public-facing. Suggested structure:

```markdown
# RetainIQ — Churn Prediction for Digital Wallets

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eliekh2/315-REPO/blob/main/notebooks/00_master_submission.ipynb)

ML pipeline for predicting customer churn in mobile money platforms.
Built on 4M Nigerian mobile money transactions and 375K customer profiles.

## Quick start

**Option 1 — Colab (recommended):** click the badge above.

**Option 2 — Local:**
\`\`\`bash
git clone https://github.com/Eliekh2/315-REPO.git
cd 315-REPO
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook notebooks/00_master_submission.ipynb
\`\`\`

## Project structure

See `CLAUDE.md` §7.

## Results

| Model | AUC | PR-AUC | Recall@10% |
|---|---|---|---|
| ... (auto-fill from benchmark.csv at submission time) |

## Authors

Elie [last name], MSBA 315, AUB Spring 2026.

## License

MIT for code. Dataset under [appropriate license].
```

---

## 9. Issues & Fixes Log (`docs/ISSUES_AND_FIXES.md`)

This is a **new deliverable** added at Phase 3.2. It documents every non-trivial issue encountered during development and how we resolved it. Two purposes:

1. **For the report (Methodology and Discussion sections):** Real ML projects have failure modes. Documenting them is what separates a research-quality paper from a tutorial. The professor specifically rewards rigor in the Optimization & Novelty section (20 pts), and showing diagnosed-and-fixed problems is rigor.
2. **For future readers / continuation work:** anyone (Elie's team, professor, future devs) opening this repo a year from now should understand WHY certain decisions were made.

### Generation procedure

Walk through the conversation history, the EDA notebook outputs, the modeling notebooks, and any markdown rationale cells. Identify every issue that warranted a course correction. For each, produce one entry in the format below.

Also prompt Elie at the start: *"Are there any issues from our chat history that you specifically want included? Any that should be excluded from the public version?"* Some issues may be too granular to surface; some may be sensitive.

### Required entries (verify each is captured if it occurred)

The following issues are known from the development phase and MUST appear in the log if they occurred. Cross-check against the actual notebook outputs before writing them up — do not invent details. If an issue did not actually occur in this project, omit it.

1. **Vendor `churn_30d` label vs engineered customer-level label disagreement (~62% agreement)** — discovered during data loading. Resolution: kept vendor label as `churned_vendor` sanity column; adopted engineered "zero transactions in May–Jun" definition as the model target. Methodology contribution.
2. **Zombie customers (~29K wallets with zero feature-window transactions)** — discovered via diagnostic queries. Initial concern: temporal contamination. Diagnosis: most are long-tenured "reactivators" who returned in May–Jun. Resolution: dropped 300 true late-registrants (registration_date > Apr 30), engineered `is_reactivator` and `dormancy_days_before_reactivation` features.
3. **Logistic Regression L1 grid search hung at 18+ minutes** — saga solver + high-cardinality OneHot of `state`/`city` → exploded design matrix. Resolution: switched solver to `liblinear`, dropped `state`/`city` from LR feature set (reintroduced for tree models), tightened convergence tolerance.
4. **Recency feature (`days_since_last_txn`) missing from LR top-15 coefficients despite literature ranking it #1** — diagnosed as multicollinearity with `txn_count_h2` (correlation > 0.9). Resolution: documented as a known linear-model limitation; tree-based models in Phase 2.2+ confirmed the literature-expected ranking.
5. **(Add any new issues that arise during Phase 2.2 onward)** — Random Forest, XGBoost, LightGBM/CatBoost, imbalance study, cluster-then-predict, SHAP all may surface their own issues. Document each as it happens.

### Entry format (use exactly this)

```markdown
### Issue N — [Short title]

**Phase:** [1.x / 2.x / 3.x]
**Date discovered:** [YYYY-MM-DD]
**Severity:** [low / medium / high / blocker]

**Symptom:**
[What was observed — error message, unexpected metric, suspicious output. Be specific.]

**Diagnosis:**
[Root cause analysis. What was actually happening? Include code snippets or output excerpts if relevant.]

**Resolution:**
[What we changed and why. Include the specific code change or config change.]

**Trade-offs / residual risk:**
[What we gave up by choosing this fix. Anything that could still bite us downstream.]

**Report citation:**
[Where this should be referenced in the research paper — e.g., "§5.2 Preprocessing", "§6.3 Limitations".]
```

### Output location

`docs/ISSUES_AND_FIXES.md` in the repo root. Commit to GitHub. Reference from the master notebook's appendix and from the research paper's methodology section.

### Sanity check

The log should have at minimum 4–6 entries by submission time. Fewer than 4 = under-documented. More than 15 = noise (some entries are too trivial to surface and should be merged or dropped).

---

## 10. Storytelling Document (`docs/PROJECT_STORY.md`)

This is a **second new deliverable** added at Phase 3.2. It is a narrative walk-through of the entire project — not technical reference material, but a written story that someone with **zero prior context** can read and understand.

### Audience

Imagine the reader is one of these three people:
- A non-technical colleague (e.g., a Whish Money product manager) who wants to understand what RetainIQ does and why it matters
- A future MSBA student inheriting this codebase
- A hiring manager Elie shows this to as a portfolio piece

The storytelling document is what they read first. The research paper is what they read if they want depth. The notebooks are what they read if they want to reproduce.

### Required structure

The document should follow this order, told as a coherent narrative — not a bulleted spec sheet. Use prose, not bullet lists, for most sections. Aim for ~3,000–5,000 words.

**Section 1 — The problem (~300 words)**
Open with the business problem. Customer churn in digital wallets — what it costs, why it matters in the MENA / Lebanese context, why Whish Money and OMT Pay specifically are worth paying attention to. Establish that early intervention is dramatically cheaper than acquisition. End with: "this project is the proof-of-concept for an intervention system."

**Section 2 — Why this dataset (~250 words)**
Explain why a Nigerian mobile money dataset was chosen as a proxy when the real target is Lebanon. Six months, 4 million transactions, 375K customers. What the dataset includes (transactions + profiles), what it doesn't (no actual cancellations), how that shaped the modeling choices. Connect to MSBA 305 lineage.

**Section 3 — How we defined "churn" (~400 words)**
This is critical. Walk through the decision: vendor's `churn_30d` was at the wrong grain and used a definition we couldn't audit. We engineered our own customer-level label: zero transactions in the last 2 months. Explain *why* this is harder than it sounds — the time-window split for leakage prevention, the reactivation pattern that emerged, why we kept both labels for cross-validation. End with the 27.2% churn rate.

**Section 4 — The architecture, in plain English (~400 words)**
Customer-grain modeling. Aggregate-first, join-profiles-second. Why that order. How the master file (375K rows × 45 cols) is the spine of everything downstream. The frozen test set policy and why it matters. What `src/config.py` and `src/evaluate.py` do. Why six experimental tracks instead of six separate projects.

**Section 5 — What we learned along the way (~600 words)**
This is the meat. Walk through the surprises and what they taught us. Pull from `ISSUES_AND_FIXES.md` but rewrite as story, not log entries. Examples of the narrative beats:

- "We expected the vendor label to agree with our engineered label more than 90% of the time. It only agreed 62%. That divergence is now a finding."
- "The first time we ran logistic regression, it hung for 20 minutes. The fix wasn't a faster computer — it was understanding why high-cardinality categoricals destabilize a specific solver, and what that taught us about feature selection for linear vs tree models."
- "Logistic regression flagged transaction frequency as the #1 churn signal. Random Forest flagged recency. Both are right — they're just looking at the data differently. This contrast became one of the most interesting findings in the report."

**Section 6 — The models, ranked (~500 words)**
Walk through each model in order. For each: what it is in one sentence, what it predicted, how well it performed, what we learned from it. End with the winner and why. Don't dump metrics — interpret them.

**Section 7 — What the model actually says about churners (~500 words)**
This is the SHAP section in narrative form. The top 5 features and what each means in business language. Show 2–3 example customers (anonymized): "Customer A is high risk because X, Y, Z. Customer B is low risk because…". Connect feature importance to business actions.

**Section 8 — The novelty contribution (~400 words)**
The cluster-then-predict approach. What it does, why it matters, how it complements the global model. Don't oversell — be honest about whether it beat the global model or merely matched it. Either outcome is a contribution.

**Section 9 — Limitations and what we'd do differently (~400 words)**
Honest discussion. The dataset is synthetic. We don't have real-world deployment validation. The label is a proxy. The reactivator pattern complicates evaluation. Future work: real Lebanese data, online learning, A/B testing for retention interventions.

**Section 10 — How to use this (~200 words)**
Brief: how to run the notebook, where the report is, how to cite it, contact info.

### Style requirements

- **First-person plural** ("we") for the team's decisions, even though Elie is solo. Sounds professional and matches academic norms.
- **Active voice.** "We engineered the label" not "the label was engineered."
- **Concrete numbers.** Don't say "churn was high" — say "27.2% of customers stopped transacting in May–June."
- **Show your work.** When a decision was hard, say so. When a result surprised us, say so.
- **No jargon without definition.** First time you use AUC, PR-AUC, SHAP, etc., define it briefly inline.
- **No code blocks.** This is the prose document. Code lives in the notebooks. The story lives here.
- **No bullet lists for narrative sections.** Bullets only for the Section 10 "how to use" list.

### Workflow

1. Generate a complete draft based on what's in the repo (notebooks, benchmark.csv, ISSUES_AND_FIXES.md, README.md).
2. Show the draft to Elie. Ask: "Anything missing? Anything overstated?"
3. Revise based on feedback.
4. Final version saved as `docs/PROJECT_STORY.md`. Commit to GitHub.

### Output location

`docs/PROJECT_STORY.md`. Linked from the repo's root `README.md` ("**📖 [Project Story](docs/PROJECT_STORY.md)** — start here if you're new to the project.").

---

## 11. Final submission checklist

Before declaring the project submission-ready:

- [ ] Master notebook (`00_master_submission.ipynb`) runs end-to-end in clean Colab session
- [ ] GitHub repo (`Eliekh2/315-REPO`) is public and accessible
- [ ] All processed data files (`customers_master.parquet`, `train.parquet`, `test.parquet`) are committed to GitHub OR have working Google Drive fallback file_ids in Cell 6
- [ ] All model `.pkl` files are committed to GitHub OR have fallback handling in Cell 7
- [ ] All hardcoded paths/URLs in the master notebook are real, not placeholders
- [ ] Colab badge in front matter and README.md works (click-tested)
- [ ] All figures in the notebook render correctly when run from Colab
- [ ] Benchmark table populates correctly
- [ ] SHAP plots generate (note: SHAP can be flaky in Colab — test thoroughly)
- [ ] Reproducibility statement is accurate (versions, seed, runtime)
- [ ] AI usage log (`docs/ai_usage_log.md`) is complete per course requirement
- [ ] Research paper PDF is committed to `docs/`
- [ ] Presentation PPTX is committed to `docs/`
- [ ] **`docs/ISSUES_AND_FIXES.md` is committed (4+ entries, all verified against actual notebook outputs)**
- [ ] **`docs/PROJECT_STORY.md` is committed (3,000–5,000 words, prose narrative, reviewed by Elie)**
- [ ] README.md links to both new documents
- [ ] CLAUDE.md §13 status tracker shows Phase 3.2 checked off

---

## 12. Common Colab gotchas (preempt these)

- **`%cd` in Colab is sticky across cells but not across kernel restarts** — always check `os.getcwd()` if anything path-related fails
- **`!pip install` inside Colab requires runtime restart** for some packages (notably tensorflow/torch) — pure ML packages here usually don't, but if `import xgboost` fails after install, restart runtime
- **`gdown` Google Drive fallback can fail with "virus scan" interstitial** for files over ~25 MB — workaround is `gdown.download(url, output, fuzzy=True, quiet=False)`; if even that fails, the file may need to be split or hosted elsewhere
- **`!git clone` in Colab is sticky to the runtime** — first run clones, subsequent runs in the same session detect the existing dir and skip; this is intentional
- **Plotly sometimes doesn't render in Colab** — stick with matplotlib + seaborn for the master notebook
- **SHAP plots in Colab need `shap.initjs()`** in the cell that produces force plots
- **CatBoost can hit a verbose-output bug in Colab** — pass `verbose=0` to all CatBoost models
- **Long cell outputs get truncated in Colab** — for the benchmark table, use `with pd.option_context('display.max_rows', None):`

---

**End of PHASE_3_2_COLAB_PACKAGING.md**
*Created: May 1, 2026. Execute when reaching Phase 3.2.*
