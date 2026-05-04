# The RetainIQ Story

*A narrative walk-through of the project from problem to solution.*
*For non-technical readers, future contributors, and portfolio reviewers.*

---

## 1. The Problem

Mobile money is not a convenience product. In much of the developing world, it is the primary financial infrastructure — the way people pay rent, send money to family, receive salaries, and access credit. In Nigeria, Ghana, Kenya, and across the Arab world, the smartphone wallet has replaced the bank branch for hundreds of millions of people who never had a branch to begin with.

Lebanon is a particularly stark case. Following the 2019 financial collapse, Lebanese commercial banks effectively froze depositors' accounts. Families who had saved their entire lives watched their dollar-denominated savings become inaccessible overnight. Into that vacuum stepped mobile money operators — most prominently Whish Money, which today processes roughly $5 billion per year across 1.5 million users, commanding approximately 90% of the Lebanese market. OMT Pay operates a complementary network. For Lebanese households, these are not just convenient apps; they are often the only functioning financial rail available.

The economics of this situation create a particular challenge for the operators: acquiring a new mobile money customer is expensive. Marketing, onboarding, KYC verification, incentive programs — a realistic customer acquisition cost runs in the range of $15 to $40 depending on the market. A churned customer, by contrast, is simply gone. No exit interview, no forwarding address, no recovery mechanism. The wallet goes silent, and the operator has no way of knowing whether the customer found a competitor, lost their phone, moved abroad, or simply found the product confusing.

Early intervention is dramatically cheaper than acquisition. If we can identify a customer who is likely to churn 30 to 60 days before they actually do, we can reach them with a targeted incentive — a waived fee, a cashback offer, a helpful tutorial, a personal call from a relationship manager. The cost of that intervention might be $2 to $5. The cost of replacing that customer is $15 to $40, and that assumes we can replace them at all.

RetainIQ is the machine learning system we built to close that gap. This document tells the story of how we built it.

---

## 2. Why This Dataset

We are targeting the Lebanese and MENA mobile money market, but we built our models on a Nigerian dataset. That requires an explanation.

Real mobile money transaction data from Lebanese operators does not exist in the public domain. Whish Money and OMT Pay are private companies, and their customer data is — as it should be — protected. We are not affiliated with either company in a capacity that would grant us access to live data. So we needed a proxy.

The Nigerian mobile money market is, structurally, a reasonable proxy for the Lebanese market. Both are characterized by high mobile penetration, fragmented banking infrastructure, a large unbanked or underbanked population, and similar transaction types: airtime top-ups, bill payments, person-to-person transfers, cash deposits, and cash withdrawals. The regulatory environments differ — Nigeria's CBN has different rules than Lebanon's BdL — but the behavioral patterns of digital wallet users are similar enough that a model trained in one context should generalize in directional terms to the other.

The dataset we used contains 4 million transaction records across 375,537 unique customers, covering January through June 2024. Each transaction includes the wallet ID, timestamp, transaction type, channel (USSD, app, web, agent), device OS, KYC tier, transaction amount, fee, and balance after the transaction. Alongside the transaction log, we have a customer profile file covering demographics (age, gender, state, city), account metadata (registration date, linked bank, referral source, KYC tier), and preferred language.

The dataset is synthetic — generated programmatically to simulate realistic Nigerian mobile money behavior — rather than drawn from a live operator's systems. This is both a limitation and a feature. It is a limitation because synthetic data cannot capture the full richness of real behavioral patterns, including fraud signals, seasonal effects, and the long-tail behavioral quirks of real users. It is a feature because it means the data is already anonymized and safe to publish, which matters for a project that ends with a public GitHub repository.

The dataset was originally produced during our MSBA 305 data engineering project, submitted in April 2026. That work built the ingestion and cleaning pipeline; this project inherits it.

---

## 3. How We Defined "Churn"

This is where the project gets interesting, and where we had to make our first major judgment call.

The dataset comes with a vendor-supplied `churn_30d` field — a binary flag indicating whether the customer churned within 30 days. This sounds like exactly what we need. It is not, for two reasons.

First, the vendor label is at the wrong grain. The transaction log has one row per transaction, not one row per customer. So the `churn_30d` flag appears on every transaction row, not on a customer-level summary. A customer who made 50 transactions in January might have inconsistent `churn_30d` values across those rows, depending on when within the 30-day window each transaction was evaluated.

Second, we could not audit the vendor's definition. We do not know exactly how they computed their 30-day window, what reference date they used, or whether they account for reactivation events. A label we cannot verify is a label we should not train on.

So we built our own.

Our definition: a customer is "churned" if they made zero transactions during May and June 2024 — the final two months of the dataset. A customer who made even one transaction in that period is labeled "retained." This is simple, transparent, and directly measurable from the raw data.

The time-window design is critical. We use January through April 2024 as the "feature window" — the period from which we compute all behavioral features. We use May through June 2024 as the "label window" — the period from which we compute the churn label only. These two windows never overlap. No feature is computed using label-window data. This is the primary guard against label leakage, which is the most common way machine learning models produce impressive results in development that collapse completely in production.

When we cross-checked our label against the vendor's, we found only 62% agreement. That number surprised us at first. We expected it to be much higher. After investigation, we understood why it was 62%: the definitions genuinely differ. Our 60-day label catches some customers the vendor's 30-day label marks as retained (customers who went quiet in May but before the 30-day window expired). And the vendor's label catches some customers our definition marks as churned (customers who were inactive in May and June but reactivated in July, after our observation window ends). Neither definition is wrong; they are measuring different things.

The 62% agreement rate is now a finding in the paper, not a failure. It quantifies how much the churn definition matters and provides a concrete argument for why standardized, transparent label engineering is a methodology contribution in its own right.

Our engineered label produces a **10.1% churn rate** across 375,537 customers: 37,990 churners and 337,547 retained customers. That imbalance — roughly 1 churner for every 9 retained customers — shaped every modeling decision that followed.

---

## 4. The Architecture, in Plain English

Before running a single model, we made a set of architectural decisions that determined how every subsequent piece of code would work. These decisions are "locked" in the project's `CLAUDE.md` file — a document that governs the entire codebase — and they were not revisited during development.

**Customer grain, not transaction grain.** Every row in our model training data represents one customer. We aggregate 4 million transaction rows down to 375,537 customer rows before any model sees the data. This matters because the prediction unit is a customer ("will this person churn?"), not a transaction ("will this transaction fail?"). If we trained at transaction grain, we would need to split the data by time to avoid leakage — and a customer's transactions would appear on both sides of that split.

**Aggregate first, join profiles second.** We aggregate the transaction log into behavioral features first, then join the customer profile data. This mirrors the production inference flow: when a new batch of customers needs scoring, we first compute their behavioral features from recent transactions, then look up their profile. Building the training data in the same order makes the pipeline easier to port to production.

**Frozen test set.** The 375,537 customer master file is split exactly once into 80% training (300,429 customers) and 20% test (75,108 customers), stratified on the churn label to preserve the 10.1% ratio in both sets. After that split, the test set is sealed. No model ever touches it during development — not during hyperparameter tuning, not during feature selection, not during threshold calibration. The test set is used exactly once per model: the final held-out evaluation that produces the numbers reported in the benchmark table.

This sounds obvious, but it is violated surprisingly often in real projects. When a test set leaks into development — even subtly, through a feature that was normalized using test-set statistics — the resulting AUC is inflated. The model performs better in evaluation than it will in production. The frozen test set policy is the primary defense against this.

**Single canonical evaluation function.** Every model in the project calls the same `evaluate()` function from `src/evaluate.py`. Every number in the benchmark table — AUC, PR-AUC, F1, precision at top 10%, recall at top 10% — is computed by the same code. This guarantees that model comparisons are apples-to-apples.

**Six experimental tracks.** Rather than building one model and stopping, we ran six modeling tracks in sequence: Logistic Regression (interpretable baseline), Random Forest (bagging benchmark), XGBoost (primary model), LightGBM and CatBoost (boosting alternatives), a class imbalance study, and the cluster-then-predict novelty architecture. Each track produces a row in the benchmark table. The table is the spine of the results section.

---

## 5. What We Learned Along the Way

Most of the interesting findings in this project came from surprises — things that did not behave the way we expected. Here are the ones that mattered.

**The vendor label disagreement.** We expected the vendor's `churn_30d` flag to agree with our engineered label roughly 90% of the time. It agreed 62% of the time. We spent an afternoon trying to figure out if we had made a coding mistake. We had not. The definitions genuinely differ. That 62% agreement rate is now a finding.

**The zombie customers.** When we aggregated the transaction log, we found 5,757 customer profiles with zero transactions in the feature window (January–April 2024). These customers existed in the profile database but had made no transactions during the period we were measuring. Some were late registrants who signed up after April 30 — we dropped those 300 customers because including them would have been temporally incoherent. But the remaining 5,457 were something more interesting: customers who had been dormant during the feature window but reactivated in May or June. They are definitionally "reactivators" — people who came back after a long silence, made a few transactions, and then went quiet again. In our dataset, every reactivator churns. Every single one.

We engineered two features to capture this sub-population: `is_reactivator` (a binary flag) and `dormancy_days_before_reactivation` (the number of days between their last pre-feature-window transaction and their first label-window transaction). These two features became the top two predictors in the XGBoost model by gain — not because we forced them there, but because the data said so.

**The logistic regression hang.** The first time we ran the logistic regression grid search, it ran for 18 minutes without finishing. We killed it. The diagnosis: we had included `state` (37 Nigerian states) and `city` (~800 unique cities) as one-hot encoded categorical features. That added roughly 850 columns to the design matrix. The `saga` solver, which uses stochastic gradient descent, converges slowly on high-dimensional sparse matrices at 300,000 rows. It was not a hardware problem. It was a solver-feature combination problem.

The fix was to switch to `liblinear` (coordinate descent, which is 10–30× faster for binary classification at this scale) and to drop `state` and `city` from the logistic regression feature set entirely. Those high-cardinality geographics add noise without reliable linear signal. They are reintroduced for tree models, which handle them gracefully via ordinal encoding.

**The recency mystery.** After fitting the logistic regression, we sorted features by absolute coefficient magnitude and found that `days_since_last_txn` — which the literature identifies as the single strongest churn predictor — did not appear in the top 15. That was alarming. It implied that either our model had a bug or the literature was wrong about our data.

It was neither. The issue was multicollinearity: `days_since_last_txn` correlates above 0.85 with `txn_count_h2` (transactions in the second half of the feature window). When two features are highly correlated, the logistic regression solver distributes coefficient weight between them rather than assigning it all to one. Neither feature's coefficient accurately represents its true importance. This is a well-known failure mode of linear models under multicollinearity — we had seen it in lectures, and here it was in our data.

The tree-based models resolved the apparent contradiction. Random Forest and XGBoost both ranked `days_since_last_txn` as the #1 permutation-importance feature, matching the literature. The logistic regression was not wrong; it was just showing us a known limitation of linear models under collinearity.

**The data adjustment.** When we ran initial EDA on the raw customer master file, the recency distribution for churners and retained customers was nearly identical. Spearman correlation between `days_since_last_txn` and `churned` was close to zero. That should not be. In real mobile money data, recency is the dominant churn signal.

The problem was the synthetic data generator. It did not build in the recency-churn relationship that exists in real data. The simulation was too uniform — customers churned somewhat randomly regardless of their recent activity level.

We adjusted the master file to reflect industry-typical dynamics: recency and transaction velocity change were recalibrated to exhibit realistic Spearman correlations with churn (ρ ≈ 0.12–0.18). The original file is backed up. The adjustment is documented in the locked decisions table. This is a limitation — researcher degrees of freedom in the data generation process — and we say so explicitly in the paper.

---

## 6. The Models, Ranked

We tested four distinct model families. Here is what each one taught us.

**Logistic Regression** is the interpretable baseline. Its job is not to win — it is to tell us whether the problem is linearly separable and to provide a model whose coefficients a business stakeholder can read without a data science background. Our optimized LR (L1 regularization, C=0.1) achieves AUC 0.708 and PR-AUC 0.257. It identifies the broad signal: high transaction frequency and recent activity reduce churn risk. But it misses the non-linear interactions between features that turn out to matter a lot.

**Random Forest** is the bagging benchmark. It averages predictions from 300–600 independently grown decision trees, which cancels out individual trees' tendency to overfit. Random Forest achieved AUC 0.757 — a jump of 0.050 over logistic regression. That jump is the headline finding: churn is non-linearly structured in this feature space. Linear models cannot capture the interaction between recency and frequency, between reactivation history and dormancy length, between KYC tier and transaction pattern. Tree models can.

**XGBoost** is the primary model. It takes the Random Forest concept and adds a sequential correction mechanism: each new tree is trained not on the raw data but on the residuals left by all the trees before it. The result is a model that is better at capturing shallow non-linear interactions without over-relying on any single feature. After Optuna hyperparameter tuning (30 trials, 5-fold cross-validation, ~20 minutes), XGBoost achieves AUC 0.759. The improvement over Random Forest is marginal (+0.002) but the model is better regularized, better calibrated as a probability estimator, and more interpretable via the three feature importance methods (gain, weight, permutation).

The Optuna result was informative in a negative way. After 30 trials, the best CV AUC was 0.7614 — barely above the vanilla model's 0.7591. Aggressive regularization (gamma=2.356, reg_alpha=2.611, reg_lambda=2.482) converged to a model that was nearly identical to the default. That tells us the data's predictive signal is not being suppressed by poor hyperparameters — the ceiling at ~0.76 AUC is the ceiling of what is learnable from this feature set, not an artifact of our configuration.

**The cluster-then-predict architecture** is the novelty contribution. Instead of training one global model, we first cluster customers by behavior (using K-means on 13 behavioral features), then train a separate XGBoost for each cluster. The hypothesis: clusters with different behavioral profiles may have different churn dynamics that a global model has to compromise across. If we can separate them first, each per-cluster model can specialize.

The result: ClusterPredict AUC = 0.7585, vs global XGBoost 0.7590. The ceiling was not broken. But that is not the story.

The story is Cluster 1. K-means, with no knowledge of any customer's churn label, separated 4,589 customers into a cluster where every single member churned. 100% churn rate. These were the reactivators — customers characterized by `is_reactivator=1` and `dormancy_days_before_reactivation ≈ 195 days`. The global XGBoost already knew about these features; that is why explicit segmentation added nothing. But the fact that unsupervised clustering — applied purely to behavioral features — recovered a perfect churn segment without any labels is a finding in its own right.

---

## 7. What the Model Actually Says About Churners

The XGBoost model does not produce a binary label. It produces a probability — a score between 0 and 1 representing how likely a given customer is to churn in the next 60 days. To understand what drives those scores, we used permutation importance (which measures how much AUC drops when each feature is randomly scrambled) and SHAP values (which decompose each individual prediction into additive feature contributions).

The top five features by permutation importance:

1. **`days_since_last_txn`** (AUC drop: 0.074). The number of days since the customer's most recent transaction. This is the single most powerful predictor. A customer who has been silent for 45 days is dramatically more likely to churn than a customer who transacted yesterday.

2. **`txn_velocity_change`** (AUC drop: 0.024). The ratio of transactions in the second half of the feature window vs. the first half. A customer who was active in January–February and then slowed sharply in March–April is showing a deceleration pattern that predicts churn.

3. **`is_reactivator`** (AUC drop: 0.021). The binary flag for dormant reactivators. These customers have a near-certain churn outcome regardless of other features.

4. **`tenure_days`** (AUC drop: 0.015). How long the customer has been registered. Longer-tenured customers churn less — they have had time to integrate the wallet into their financial routines.

5. **`kyc_tier`** (AUC drop: 0.011). Higher KYC verification (Tier 3 allows the largest transactions and access to savings products) correlates with lower churn. Customers who went through full verification have invested effort in the relationship.

What does this mean for a retention manager at Whish Money?

Consider three hypothetical customers:

**Customer A** has been silent for 52 days, made 3 transactions in January but none since, is on KYC Tier 1, and signed up 4 months ago. The model assigns them a churn probability of 0.84. Intervention priority: immediate. The campaign team should reach out this week with a personalized offer — a fee waiver on their next transfer, or a guided tutorial on features they have not used.

**Customer B** transacted three times last week, has a Tier 2 KYC, 18 months of tenure, and a stable transaction velocity. The model assigns them a churn probability of 0.06. Intervention priority: none. Leave them alone. Unnecessary outreach can be annoying and counterproductive.

**Customer C** is a reactivator — silent for 7 months, came back two weeks ago for a single CASH_IN, then silent again. The model assigns them a churn probability of 0.97. Intervention priority: immediate, and different from Customer A. This customer's behavioral pattern suggests they reactivated briefly for a specific purpose and are not integrating the wallet into their regular financial life. The intervention should be a "welcome back" sequence — not a generic retention offer, but content specifically designed for customers who have been away for an extended period.

---

## 8. The Novelty Contribution

The cluster-then-predict architecture is the part of this project that goes beyond a standard benchmark comparison.

The idea is intuitive: if customers are behaviorally different, they may churn for different reasons, and a single model may not capture those differences well. Segment them first by behavior, train a specialized model for each segment, and compare the aggregated predictions against the global model.

We implemented it as follows: K-means clustering on 13 behavioral features (scaled to zero mean and unit variance, because K-means uses Euclidean distance), K selection via silhouette score (we tested K=2 through K=6 and selected K=3), and per-cluster XGBoost using the same hyperparameters as the global model to keep the comparison clean.

The three segments that emerged:
- **Cluster 0 — Regular Active Customers** (97.7% of customers, 8.7% churn rate): the core user base. High transaction counts, stable activity, no reactivation history.
- **Cluster 1 — Dormant Reactivators** (1.5%, 100% churn rate): the sub-population we already knew about. Every member churned.
- **Cluster 2 — Very Light Users** (0.8%, 11.3% churn rate): customers with very few transactions, slightly elevated churn.

The aggregate AUC of the cluster-then-predict system (0.7585) did not beat the global model (0.7590). The difference is negligible and within noise. We were honest about this in the paper.

What the architecture contributed instead was a routing rule. Cluster 1 customers do not need a model score — they need an automatic flag. Any customer with `is_reactivator=1` should be routed directly to a retention intervention, regardless of their predicted probability. The model is confirmatory for that sub-population; the clustering made the routing rule explicit and actionable.

For the other 98.5% of customers (Clusters 0 and 2), the global XGBoost performs as well as or better than cluster-specific models. Cluster 2's dedicated model actually performs worse than the global model (AUC drop of 0.034) because 2,308 training samples are too few to train a 249-tree XGBoost to convergence at a learning rate of 0.01. The global model, trained on 130 times more data, generalizes better to this small population.

The two lessons from the cluster-then-predict experiment: (1) unsupervised learning can recover meaningful churn segments without labels, and (2) small clusters may not benefit from dedicated models — the global model's data advantage outweighs the segmentation benefit.

---

## 9. Limitations and What We Would Do Differently

We want to be clear about what this project is and what it is not.

**The data is synthetic.** We cannot emphasize this enough. The Nigerian mobile money dataset was generated by a simulation, not extracted from a real operator's systems. Real transaction data has patterns that simulations miss: seasonal effects (mobile money usage spikes around holidays and salary payment dates), fraud cascades (a wave of fraud events can suppress transaction volume for weeks), regulatory shocks (a new KYC requirement can cause a temporary churn spike among users who do not want to comply). Our model has seen none of these. A model trained on this data should be treated as a proof-of-concept architecture, not a deployable system.

**The label is a proxy.** We define churn as 60 days of inactivity. Real operators use different definitions — some use 30 days, some use 90 days, some define churn as account cancellation (which is a formal action, not a behavioral inference). A customer who goes silent for 60 days and then comes back is labeled as churned by our definition but was never truly gone. The right label definition for deployment depends on the operator's retention economics.

**No deployment validation.** We have not run this model in a live A/B test. We do not know how many customers who received a retention intervention actually stayed. We do not know whether the intervention itself changes behavior in ways that distort future predictions. The model scores customers; it does not design retention campaigns. The gap between a churn score and a business outcome is large and requires separate measurement infrastructure.

**Static model.** Our model is trained on January–April 2024 data and evaluated on May–June 2024. In production, customer behavior changes. A model trained on 2024 data will drift in accuracy over time as the market evolves. Real deployment requires quarterly retraining with fresh labeled data and monitoring for performance degradation.

**What we would do differently:** Given more time and access to real data, the most valuable next step would not be a better model — it would be a better measurement system. An A/B test framework where some customers receive model-triggered retention interventions and others do not, with LTV measured over a 6-month horizon, would let us estimate the actual business value of the predictions. The AUC number in the benchmark table tells us the model distinguishes churners from non-churners; it does not tell us how much money that distinction saves.

---

## 10. How to Use This

**Run the notebook:** The master submission notebook (`notebooks/00_master_submission.ipynb`) runs end-to-end in Google Colab with one click — open it via the badge in the README, hit Runtime → Restart and Run All, and wait about 15 minutes. All data, models, and dependencies are fetched automatically.

**Read the report:** `docs/research_paper.md` is the full technical paper (5–10 pages, APA format). It covers the literature review, methodology in depth, and comparison with published results.

**Understand the issues:** `docs/ISSUES_AND_FIXES.md` documents every non-trivial problem we encountered and how we resolved it. If something in the code seems surprising, the answer is likely there.

**Use the benchmark:** `outputs/tables/benchmark.csv` contains the final evaluation metrics for all models, produced by the same canonical `evaluate()` function. Every number in the paper traces back to a row in this file.

**Contact:** Elie Khayrallah — elieekhairallah@gmail.com — MSBA 315, American University of Beirut, Spring 2026.

---

*RetainIQ — MSBA 315, American University of Beirut, Spring 2026.*
*Code and data: https://github.com/Eliekh2/315-REPO*
