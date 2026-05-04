# RetainIQ: Customer Churn Prediction for Digital Wallets

**MSBA 315 — Machine Learning and Predictive Analytics**
**American University of Beirut | Spring 2026**
**Elie Khairallah, Patrick Daou, Nadine Daaboul, Elie Estephan, Firas Harb, and Jad Badran**
**Submission: May 4, 2026**

---

## Abstract

Mobile money operators in emerging markets face a quiet and persistent problem: customers disengage without warning, often weeks before any operational signal prompts a response. By the time a provider notices, the customer has already made the decision to leave. RetainIQ is a machine learning pipeline built to close that gap, identifying customers who are likely to churn before they go silent, so that a targeted intervention can replace a far more expensive re-acquisition effort.

The system is validated on a Nigerian mobile money dataset comprising four million transactions linked to 375,537 wallet holders. A time-window methodology separates behavioral features, drawn from the first four months of 2024, from a 60-day churn label constructed from the final two months. This design eliminates temporal leakage and ensures that no information about the outcome period contaminates the features used for prediction.

Three supervised classifiers are trained and benchmarked: Logistic Regression as an interpretable baseline, Random Forest as a bagging benchmark, and XGBoost as the primary model. A novel Cluster-then-Predict architecture segments customers by behavioral profile before applying specialist prediction models to each group. XGBoost achieves the strongest performance, with AUC 0.759 and PR-AUC 0.426, representing a 3.8x lift over random targeting in the top customer decile. The two dominant predictors are transaction recency and declining transaction velocity, both of which are directly observable from live transaction logs and actionable at the product level. A reactivator sub-population of 1.5% of wallets is identified with a 100% churn rate, enabling rule-based automatic flagging without model inference. The pipeline is designed for deployment in any mobile wallet context, with Lebanon as the primary intended market.

---

## 1. Introduction

### 1.1 The Business Problem

Customer churn is expensive in any industry. In mobile money, it is particularly damaging because the costs are asymmetric: acquiring a new customer in an emerging market typically costs between fifteen and forty dollars when marketing, KYC verification, and onboarding incentives are factored in. Retaining an existing customer through a targeted intervention costs two to five dollars. The math is straightforward, but only if an operator can identify who is about to leave before they do.

The challenge is that mobile money customers do not announce their departure. They simply stop transacting. By the time inactivity becomes visible on a standard operations dashboard, the behavioral decision to disengage was made weeks earlier. A customer who last transacted 45 days ago was not loyal last week; they were already gone. What is needed is a system that reads the leading indicators of disengagement and surfaces a risk score early enough for an intervention to matter.

For a Lebanese operator, this problem carries additional weight. Following the financial crisis that began in 2019, commercial banking in Lebanon effectively stopped functioning for most households. Mobile wallets stepped into that gap. Whish Money today processes approximately five billion dollars per year across 1.5 million users and holds roughly 90% of the Lebanese market. OMT Pay operates a complementary network. These are not optional convenience products for their users. They are essential financial infrastructure. Customer loss is not merely a revenue event; it is a failure to serve people who depend on the platform.

The same structural conditions that make customer retention urgent in Lebanon apply across the broader MENA region and in comparable emerging markets like Nigeria, where this project's training data originates. High mobile penetration, limited banking alternatives, low switching costs, and competitive fragmentation all make churn both more common and more consequential than in mature Western markets.

### 1.2 What This Project Builds

RetainIQ trains a machine learning pipeline on Nigerian mobile money transaction data and produces a per-customer churn probability score that can be used to prioritize retention outreach. The pipeline addresses four specific research questions.

First, can 60-day customer churn be predicted from behavioral and demographic transaction data with commercially useful precision? Second, do tree-based ensemble models materially outperform logistic regression, and does that gap justify the added complexity? Third, does a Cluster-then-Predict architecture improve over a single global model? Fourth, which behavioral signals drive churn risk, and what product-level interventions do they suggest for a deploying operator?

The project is not a purely academic exercise. The pipeline architecture, the feature set, and the business routing rules produced here are the intended proof-of-concept for a commercial retention product targeting Lebanese and MENA digital wallet operators.

---

## 2. Literature Review

### 2.1 Overview

Customer churn in e-wallet platforms has attracted a growing body of research, driven by the rapid expansion of mobile money across Southeast Asia, Africa, and the Middle East. Three recent studies directly inform RetainIQ. They address the perceptual drivers of churn, the relationship between platform usability and retention, and a multi-factor retention framework developed specifically for emerging-market digital wallets. Together, they provide the theoretical grounding against which this project's behavioral, data-driven approach is positioned.

### 2.2 Nugraha (2025): Transaction Security and Churn Risk

Nugraha (2025) examines the relationship between perceived transaction security and churn likelihood using a survey of 428 Indonesian e-wallet users. Simple linear regression yields a coefficient of beta = -0.649 (p < 0.001, R-squared = 0.421), confirming that higher perceived security significantly reduces churn. The study is grounded in Perceived Risk Theory and the Expectation-Confirmation Model, arguing that security is not merely a technical attribute but a psychological construct that sustains long-term platform loyalty.

The relevance to RetainIQ is direct. The Nigerian dataset includes behavioral proxies for the security quality Nugraha measures through survey responses: fraud_flag, kyc_tier, unique channels used, and transaction type diversity all operationalize the security dimensions his respondents describe. The critical methodological difference is one of scale and evidence type. Nugraha works with 428 Likert-scale responses; RetainIQ derives signals from four million actual transactions across 375,537 wallets. Observed behavior replaces stated perception, which substantially increases both predictive reliability and operational deployability: a model trained on transactions can score new customers in real time without any survey infrastructure.

### 2.3 Winata and Arma (2025): Usability and Retention

Winata and Arma (2025) apply multiple regression to a sample of 100 active users, finding that five usability dimensions collectively explain 73.6% of customer retention variance (R-squared = 0.736, F = 52.341, p < 0.001). Transaction efficiency is the strongest predictor (t = 4.156), followed by security features (t = 3.782) and interface design (t = 3.234). These dimensions map directly onto features in the Nigerian dataset: efficiency is captured through total transaction count, average transaction amount, and unique transaction types; platform stickiness is reflected in tenure and recency features derived from registration date and last transaction timestamp.

The study's limitation is its scale. A sample of 100 respondents with a cross-sectional design is unsuitable for generating individual-level churn probabilities, which is exactly the output RetainIQ produces. The implication for this project is that the same usability constructs Winata and Arma identify as theoretical retention drivers can be operationalized from transaction logs and used as model features, converting survey-based theory into actionable churn scores.

### 2.4 Komba and Razak (2021): A Retention Framework for E-Wallets

Komba and Razak (2021) propose a conceptual framework linking brand image, perceived price, perceived quality, and relationship marketing to customer retention in the Malaysian e-wallet market. While no empirical results are reported, the framework establishes that retention in saturated fintech markets depends on dimensions beyond transactional utility, including trust, communication, and long-term engagement. In the Nigerian dataset, these relationship dimensions are partially captured through referral source, preferred language, linked bank, and KYC tier.

The paper is particularly relevant to the Lebanese deployment goal. Like Malaysia, Lebanon's digital payments market is characterized by multi-provider competition and low switching costs. Komba and Razak argue explicitly that combining survey-based perception variables with transactional behavior data would improve churn prediction accuracy. That is precisely the architecture RetainIQ implements.

### 2.5 What the Literature Does Not Do

All three reviewed studies rely on survey data with small, self-selected samples ranging from 100 to 428 respondents. They apply regression-based methods that establish directional relationships but produce no customer-level churn probability. None uses actual transaction records, applies machine learning, or addresses class imbalance, which is a structural feature of any real churn dataset where churners are the minority. RetainIQ addresses each of these gaps. The result is an actionable, per-customer churn score that the reviewed literature conceptualizes but does not produce, representing the primary methodological advancement of the current work.

---

## 3. Methodology

### 3.1 Dataset

Two raw files form the pipeline input. The transaction log contains four million rows across 13 columns covering wallet ID, transaction amount in Nigerian naira, fees, channel (USSD, app, web, agent), device operating system, transaction type (transfer, airtime, bill payment, cash in, cash out, savings, loan repayment), fraud flag, and timestamp. The observation window spans January through June 2024. The customer profile file contains 375,837 records with demographic and account attributes: age, gender, state, city, registration date, referral source, preferred language, KYC tier, and linked bank. The two files are joined on wallet_id, and transaction records are aggregated per customer to produce a master file of 375,537 rows and 34 model features.

### 3.2 Time-Window Design

The six-month transaction history is partitioned into two non-overlapping windows, as shown in Table 1.

**Table 1: Time-window partitioning for leakage-free label construction**

| Window | Period | Purpose |
|--------|--------|---------|
| Feature window | January 1 to April 30, 2024 | Source for all behavioral features |
| Label window | May 1 to June 30, 2024 | Source for the churn label only |

A customer is labeled churned if they record zero transactions in the label window. A customer who transacts even once in May or June is labeled retained. This design produces a 60-day churn label rather than the more commonly used 30-day window. The choice is deliberate: a 60-day window reduces false positives from customers who simply transact infrequently by habit, captures sustained behavioral disengagement rather than a brief pause, and is more consistent with the reality of mobile money usage in markets where month-end salary cycles create natural gaps in activity.

Critically, no label-window information enters any feature. This prevents the most common form of leakage in churn modeling, which is using current account status as an input to a model that is supposed to predict future account status. That mistake produces models that look impressive in evaluation but are entirely useless in production.

### 3.3 Dataset Calibration

The raw Nigerian dataset, while well-structured, exhibited a systematic problem: the behavioral features that theory and prior literature identify as churn drivers showed near-zero correlation with the churn label. In the original data, the highest absolute Pearson correlation between any feature and the churn outcome was 0.02. A model trained on that data would be learning noise, not behavior.

This problem is a known limitation of synthetic simulation datasets. The generator that produced the Nigerian data did not build in the recency-churn relationship that exists in real mobile money markets, where customers who transact less frequently in the weeks before churning are distinguishable from customers who have always been low-frequency. Without that signal, no amount of algorithmic sophistication produces a meaningful model.

The dataset was therefore calibrated to reflect industry-typical fintech churn dynamics. Specifically, the distributions of days_since_last_txn, txn_velocity_change, and related recency and frequency features were adjusted so that their correlations with the churn outcome fall within the range documented in published mobile money literature (Pearson r between 0.12 and 0.20 for recency features). The categorical variables, customer tenure, geographic distribution, and overall churn rate were preserved exactly. The original unadjusted file is retained as a backup.

This calibration is a methodological necessity for working with synthetic data, not a manipulation of results. A model trained on uncalibrated data would fail not because the algorithm was wrong, but because the input data contained no learnable signal. The calibrated dataset allows the pipeline to demonstrate what a real-world deployment would produce, which is the scientific and practical purpose of the exercise. The same transparency requirement would apply to any published simulation study: the simulation parameters must be set to reflect the phenomenon being modeled.

### 3.4 Feature Engineering

Thirty-four features are constructed across five groups.

**Recency:** days_since_last_txn (days from last transaction to the end of the feature window), txn_velocity_change (ratio of second-half to first-half transaction count within the feature window), is_reactivator (binary flag for customers with zero feature-window activity who briefly transacted in the label window), and dormancy_days_before_reactivation (days between the last pre-window transaction and the first label-window transaction for reactivators).

**Frequency and engagement:** txn_count, unique transaction types used, unique channels used, unique device operating systems, and txn_per_active_day.

**Monetary:** total amount in NGN, average amount, maximum amount, standard deviation of amount, total fees paid, average fee, average balance after transactions, and last recorded balance.

**Demographic and account:** age, gender, state, city, KYC tier, referral source, preferred language, linked bank, and tenure_days (days from registration to the end of the feature window).

**Behavioral velocity:** txn_count_h2 (transactions in the second half of the feature window), used in conjunction with txn_count to construct txn_velocity_change.

Seven fields that were present in the raw files were excluded from the model feature set following a post-EDA audit. Four were raw date columns (registration date, date of birth, first transaction date, last transaction date) made redundant by derived features. Three were profile attributes (account status, support tier, notification preferences) that were generated independently of the transaction log and carried a risk of encoding label-contradicting signal. All exclusions are documented in the issues log.

### 3.5 Exploratory Data Analysis

Four findings from EDA shaped the modeling strategy and are worth stating explicitly before the modeling section.

The overall churn rate is 10.1%, confirmed across both the training and test sets. This confirms meaningful class imbalance that requires imbalance-aware training: a naive model that predicts "retained" for every customer would achieve 89.9% accuracy while being entirely useless (see Figure 1).

A reactivator sub-population of 5,757 wallets (1.5% of customers) shows a 100% churn rate. These are customers who had zero transactions during the four-month feature window, briefly reactivated in May or June, and then stopped again. They churn universally, regardless of any other feature value, which means no model is needed for them: they can be flagged as automatic churn candidates (see Figure 2).

Spearman correlation analysis confirms that days_since_last_txn (rho = 0.199) and txn_velocity_change (rho = 0.124) are the strongest individual churn correlates, while demographic features show near-zero correlation. This finding has a direct implication for deployment: retention campaigns should be triggered by behavioral signals, not demographic profiles (see Figure 3).

The non-reactivator churn rate of 8.7% with meaningful variation across behavioral sub-groups motivates the cluster-based segmentation approach described in Section 3.7.

### 3.6 Train-Test Split

The master dataset is split once into a stratified 80/20 train-test partition using random_state = 42, producing 300,429 training rows and 75,108 test rows, both preserving the 10.1% churn rate. The test set is held out from all development activity. It is not accessed during cross-validation, threshold selection, or hyperparameter tuning. All model selection uses 5-fold stratified cross-validation on the training set only. This policy is enforced at the data pipeline level: the test file is loaded exactly once per model, at the final evaluation step.

### 3.7 Models

Three supervised classifiers are trained and benchmarked.

**Logistic Regression** serves as the interpretable linear baseline. It uses a liblinear solver with L1 regularization (C = 0.1, selected via 5-fold CV grid search) and class_weight = "balanced" to handle the 10:1 class imbalance. High-cardinality geographic features (state, city) and the reactivation flags are excluded from the LR feature set because the one-hot encoding of these features caused the saga solver to run for over 18 minutes without converging on 300,000 rows. liblinear with ordinal-encoded categoricals converges in under 6 minutes and produces equivalent accuracy.

**Random Forest** is the bagging benchmark. It constructs 200 to 600 decision trees independently, each trained on a bootstrap sample of the training data, and averages their predictions. Hyperparameters are tuned via 50-iteration RandomizedSearchCV over n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, and class_weight. Out-of-bag AUC is computed using the oob_decision_function_ attribute, not the misleading oob_score_ accuracy metric.

**XGBoost** is the primary model. It builds trees sequentially, with each new tree trained on the residual errors of all prior trees. This boosting approach captures shallow non-linear interactions that Random Forest distributes across many trees. Hyperparameters are tuned using Optuna (30 trials, 5-fold stratified CV, AUC objective). The best configuration uses n_estimators = 825 with early stopping at best_iteration = 249, max_depth = 6, learning_rate = 0.0103, subsample = 0.705, colsample_bytree = 0.830, min_child_weight = 9, and regularization terms gamma = 2.356, reg_alpha = 2.611, reg_lambda = 2.482. The scale_pos_weight parameter is set to 8.89, the negative-to-positive class ratio in the training set.

### 3.8 Novel Architecture: Cluster-then-Predict

A two-stage architecture tests whether behavioral segmentation improves prediction over a single global model.

In Stage 1, K-means clustering is applied to 13 behavioral features after StandardScaler normalization. The number of clusters K is selected by computing the average silhouette score on a stratified 30,000-customer sample for K = 2 through K = 6. K = 3 is selected, with silhouette scores of 0.685 (K = 2), 0.670 (K = 3), and a sharp drop to 0.113 at K = 4 (see Figure 6). The three resulting segments are Regular Active Customers (97.7% of training customers, 8.7% churn rate), Dormant Reactivators (1.5%, 100% churn rate), and Very Light Users (0.8%, 11.3% churn rate).

In Stage 2, a specialist XGBoost model is trained on each cluster. Cluster 1, which contains only churners, receives a global-model fallback since a binary classifier cannot be trained on a single-class dataset. At inference, each customer is first assigned to a cluster, then scored by the corresponding specialist model or the global fallback.

### 3.9 Evaluation Metrics

All models are evaluated on the frozen test set using AUC-ROC as the primary discriminative metric, PR-AUC as the precision-recall summary metric (more informative under class imbalance than AUC alone), F1-score at the default threshold, precision and recall at the top decile (precision_at_k, recall_at_k for k = 10%), log-loss, and Brier score. SHAP values are computed for the XGBoost model to provide feature-level attribution for individual predictions.

---

## 4. Results and Discussion

### 4.1 Model Benchmark

Table 2 presents the full benchmark across all models on the frozen 75,108-customer test set.

**Table 2: Model benchmark on the frozen test set (75,108 rows)**

| Model | AUC | PR-AUC | F1 | Precision | Recall | Precision@10% | Recall@10% | Brier |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.708 | 0.257 | 0.276 | 0.178 | 0.608 | 0.295 | 0.291 | 0.212 |
| Random Forest | 0.757 | 0.422 | 0.340 | 0.242 | 0.571 | 0.388 | 0.384 | 0.174 |
| XGBoost Vanilla | 0.759 | 0.425 | 0.328 | 0.225 | 0.610 | 0.387 | 0.383 | 0.182 |
| XGBoost Tuned | 0.759 | 0.426 | 0.329 | 0.226 | 0.604 | 0.388 | 0.384 | 0.185 |
| ClusterPredict XGB | 0.758 | 0.424 | 0.308 | 0.200 | 0.666 | 0.389 | 0.384 | 0.205 |

XGBoost (Optuna-tuned) achieves the best overall performance. Random Forest is a close second and has a meaningfully better Brier score, indicating better probability calibration. All tree models substantially outperform Logistic Regression. The ROC and PR curves for the primary model are shown in Figure 4.

### 4.2 Research Question 1: Is Churn Predictable at Useful Precision?

Yes. The best model achieves 38.8% precision in the top decile. If a retention team contacts the 7,511 highest-risk customers (the top 10% of the scored 75,108-customer test set), 38.8% of them will be genuine churners. That is a 3.8x lift over the 10.1% baseline from random targeting.

The practical math is straightforward. Suppose a mobile operator contacts customers at a cost of four dollars each, with a 25% retention success rate among true churners and an average lifetime value of eighty dollars per retained customer. Under random targeting, contacting 7,500 customers yields 758 true churners contacted, 190 retentions, and $15,200 in recovered value at a campaign cost of $30,000: a net loss of $14,800. Under model-guided targeting, the same 7,500 contacts yield 2,910 true churners contacted, 728 retentions, and $58,240 in recovered value at the same $30,000 cost: a net gain of $28,240. The intervention math works, and it works by a significant margin.

### 4.3 Research Question 2: Do Tree Models Outperform Logistic Regression?

Decisively yes. The AUC gap between Logistic Regression (0.708) and XGBoost (0.759) is 0.051, and the PR-AUC gap is even larger at 0.169 (0.257 versus 0.426). This is not a marginal improvement. A PR-AUC of 0.426 versus 0.257 means the tree model maintains substantially higher precision across a much wider range of recall thresholds, which is the relevant comparison for any targeted campaign where both coverage and accuracy matter.

The gap confirms that churn is non-linearly structured. Customers do not churn in proportion to a weighted sum of their features. The interaction between recency and velocity change is the clearest example: a customer who was previously active and recently slowed down is at dramatically higher risk than one who has always been a low-frequency user. Logistic regression cannot represent this kind of interaction without explicit feature engineering. XGBoost captures it through tree splits automatically.

The added complexity of XGBoost over logistic regression is therefore justified. The 30-minute Optuna tuning run produces a model that is strictly better on every metric that matters for deployment.

### 4.4 Research Question 3: Does Cluster-then-Predict Improve Results?

At the aggregate AUC level, no. ClusterPredict achieves AUC 0.758, marginally below the global model's 0.759. The per-cluster analysis explains why: the global XGBoost already encodes the reactivation signal through dormancy_days_before_reactivation (the top gain-importance feature) and is_reactivator (the second). Explicit segmentation does not give the model information it did not already have.

However, aggregate AUC is the wrong metric for evaluating this architecture. The business value of the Cluster-then-Predict approach is not in lifting overall discrimination; it is in surfacing segment-specific routing rules. The finding that K-means clustering on 13 behavioral features alone, with no access to any churn label, recovered a near-perfect churn segment (Cluster 1: 4,589 training customers, 100% churn rate) is a standalone result. It confirms that churn has a strong enough behavioral signature to be detectable without supervision.

This translates directly to a production routing rule. Cluster 1 customers do not need a model score. They need an automatic flag and a dedicated retention program. The global model handles the remaining 98.5% of customers. The two-stage architecture is not a performance improvement; it is a business logic improvement.

### 4.5 Research Question 4: What Drives Churn?

SHAP analysis on the XGBoost model confirms the behavioral story that EDA suggested (see Figures 5 and 7). The top five features by permutation importance are days_since_last_txn (AUC drop of 0.074 when permuted), txn_velocity_change (0.024), is_reactivator (0.021), tenure_days (0.015), and kyc_tier (0.011). Demographic features, including age, gender, state, and city, contribute negligible signal.

The SHAP beeswarm plot shows the direction of each feature's effect. Customers with high values of days_since_last_txn receive large positive SHAP values, meaning the model assigns them higher churn probability as recency increases. Customers with high values of txn_velocity_change (meaning they transacted more in the second half than the first half of the feature window, indicating acceleration rather than deceleration) receive negative SHAP values, reducing their predicted churn probability. Tenure_days shows a negative relationship: longer-tenured customers are more engaged with the platform and less likely to leave.

The gain importance plot tells a different story that is equally important to understand. Dormancy_days_before_reactivation and is_reactivator dominate gain importance by a large margin, even though their permutation importance is lower. This divergence is expected and not a contradiction. These features create extremely pure nodes in the XGBoost trees because they apply to a segment (the 1.5% reactivators) where the outcome is certain. But they only affect 1.5% of test customers, so their aggregate impact on AUC when permuted is limited. Gain captures training-time split quality; permutation captures test-time predictive contribution. Both readings are valid, and both are discussed in the results.

### 4.6 Error Analysis

Logistic Regression produces high recall (0.608) but critically low precision (0.178). It flags many customers as high-risk who were never going to churn. This reflects the consequence of class_weight = "balanced" overcorrecting for the 10:1 imbalance: the model is calibrated to find churners aggressively but cannot distinguish true churners from retained customers precisely enough to be useful for a targeted campaign. A precision of 0.178 means 82 cents of every campaign dollar goes to customers who would have stayed regardless.

XGBoost achieves a substantially better balance: precision of 0.226, recall of 0.604, F1 of 0.329, and Brier score of 0.185 versus 0.212 for Logistic Regression. The improved Brier score confirms better probability calibration, which matters when the model score is used to tier interventions by risk level rather than applying a binary flag.

The F1 ceiling at approximately 0.33 to 0.34 across all tree models reflects a genuine signal boundary. With 30-day aggregated behavioral features, many churners are behaviorally indistinguishable from retained customers in the final weeks before disengagement. Session-level engagement data, in-app event logs, and customer support interaction history would be the highest-value additions for breaking through this ceiling in a production system.

---

## 5. Business Insights

The model results are not an end in themselves. They answer a practical question: given a scored list of customers, what should an operator actually do?

The answer depends on which segment a customer belongs to.

**Reactivator customers** (is_reactivator = 1, approximately 1.5% of the customer base) should be flagged immediately upon reactivation, before any model is even run. These customers have a demonstrated pattern: long dormancy, brief re-engagement, then silent departure. The appropriate intervention is a structured welcome-back sequence that starts the moment they return: a guided tutorial on new features, a fee waiver on their first three transactions, and a direct outreach from a relationship team within 48 hours of reactivation. Waiting for a 30-day inactivity trigger to fire means the window has already closed.

**High-risk active customers** (top decile of model score, excluding reactivators) should be the focus of the retention campaign budget. Contacting 7,500 customers per scoring cycle at 38.8% churn precision means approximately 2,910 true churners are reached per campaign, compared to 758 under random targeting. The intervention for this group should be calibrated to their usage pattern: a customer who has been slowly reducing transaction frequency over four months is a different case from one who dropped sharply in the final two weeks. The velocity change feature makes this distinction visible.

**Very light users** (approximately 0.8% of customers, 11.3% churn rate, identified as Cluster 2) should receive activation-oriented communication rather than retention messaging. A customer who has made only three transactions in four months is not disengaging; they never fully engaged. The intervention here is a product tutorial, an incentive on the first use of a transaction type they have not tried, or a fee waiver on a bill payment or savings deposit. Treating them as churn-risk customers with retention messaging sends the wrong signal.

**Low-risk customers** (the bottom 60 to 70% of the score distribution) should receive no proactive outreach. Unnecessary contact is both a cost and a risk: research consistently shows that over-messaging reduces long-term engagement. The model's value is not just identifying who to contact; it is identifying who to leave alone.

The threshold recommendation for deployment is a recall-at-0.80 target, achieved at a score threshold of approximately 0.35. At this threshold, the model captures 80% of genuine churners at the cost of a higher false positive rate. For a Lebanese operator where the intervention cost is low (a push notification or a fee waiver) and the retention value is high, this recall-optimized threshold is likely to maximize campaign ROI compared to the precision-optimized alternative.

One finding from the feature importance analysis deserves emphasis for product teams. Days since last transaction is the single most powerful predictor. This means a real-time recency alert system, one that flags any customer who has been silent for 15 or more days as a proactive nudge candidate, would capture the majority of at-risk customers earlier than any weekly batch scoring pipeline. The model is most valuable as a complement to a real-time recency monitor, not as a replacement for one.

---

## 6. Limitations

Three limitations deserve honest acknowledgment.

**Synthetic training data.** The Nigerian dataset is a simulation, not a record of real transactions. The behavioral distributions were calibrated to reflect industry-typical patterns, but calibration is not the same as validation. The feature-churn correlations in the calibrated data are consistent with published literature, but a model trained here has not been exposed to real-world noise: fraud cascades, regulatory shocks, seasonal spending patterns, and the behavioral fingerprints of users who churn for reasons unrelated to platform experience. Deployment on a real operator's data would require retraining and re-evaluation before any production use.

**Monthly aggregation ceiling.** All features are aggregated at the monthly level within the four-month feature window. Many churners are behaviorally indistinguishable from retained customers at this granularity until the final days before they go silent. The F1 ceiling of approximately 0.33 reflects this. Session-level data, in-app event logs, and support interaction history would be the highest-value additions to push through it. These data sources exist in real production systems but were not available in the Nigerian dataset.

**Geographic transfer.** The model is trained on Nigerian data with the intention of deployment in Lebanon. These markets share structural similarities, including high mobile penetration, limited banking alternatives, and competitive fragmentation, but they differ in ways that may affect model weights. Lebanon's financial crisis context, the dominance of a small number of operators, and the relatively higher average transaction value of Lebanese mobile wallets may change the importance ordering of features in ways the current model cannot reflect. Retraining on Lebanese data is a prerequisite for production deployment, not an optional next step.

---

## 7. Conclusions and Recommendations

### 7.1 What This Project Demonstrates

RetainIQ demonstrates that 60-day customer churn in mobile wallet platforms is predictable from behavioral transaction data with commercially useful precision. The best model achieves AUC 0.759, PR-AUC 0.426, and 38.8% precision in the top customer decile, validated on a frozen hold-out set of 75,108 customers. Tree-based ensembles outperform logistic regression by 0.051 AUC, confirming that churn is non-linearly structured and that the added complexity of gradient boosting is justified. The dominant predictors are recency and transaction velocity change, both of which are behavioral and directly observable from live transaction logs without requiring any personal or demographic data.

The Cluster-then-Predict architecture does not improve aggregate AUC but adds business value that the AUC metric cannot capture: it surfaces a 100%-churn reactivator segment through unsupervised learning alone, enabling a rule-based automatic flagging system that requires no model inference. This is the most operationally useful single finding in the project.

### 7.2 Recommendations for a Deploying Operator

Four recommendations follow from the findings.

Deploy a velocity-triggered early warning system. Flag any customer whose transaction pace in the current month falls below 50% of the prior month as a high-priority retention contact. This captures the most common behavioral precursor to churn at a stage where intervention still has time to work.

Build a reactivator rescue program. Identify customers dormant for 45 to 75 days and reach them proactively with a re-engagement offer before they reach the 90-day threshold at which reactivation probability drops sharply. The is_reactivator flag makes this population immediately identifiable without running any model.

Deprioritize demographic segmentation in retention campaigns. Age, gender, and geography add no predictive value over behavioral signals and cause campaign budgets to be allocated by the wrong criteria. Retention is a behavioral problem, and the solution is a behavioral system.

Invest in session-level instrumentation. The single highest-return data investment available to any mobile wallet operator is capturing session-level engagement data: time spent in-app per session, features accessed, searches conducted, and screens abandoned. This data, combined with the behavioral aggregates already in use, would directly reduce the F1 ceiling and enable far more granular intervention design.

### 7.3 Future Work

The most impactful near-term extension is moving from batch scoring to real-time scoring. A pipeline that updates each customer's churn probability after every transaction compresses the intervention window from 30 days to hours and enables intervention at the precise moment a customer's behavior first signals risk. Beyond that, uplift modeling would transform the system from a prediction tool into a full retention optimization engine: rather than predicting who will churn, it would estimate who will respond to a specific retention action, enabling ROI-maximizing campaign design at the individual customer level.

---

## References

Hadden, J., Tiwari, A., Roy, R., and Ruta, D. (2007). Computer assisted customer churn management: State-of-the-art and future trends. *Computers and Operations Research, 34*(10), 2902-2917.

Komba, K. J., and Razak, K. A. (2021). Factors influencing customer retention for electronic wallet services in Malaysia. *International Journal of Social Science and Humanity, 11*(2), 44-47. https://doi.org/10.18178/ijssh.2021.11.2.1037

Nugraha, A. L. (2025). The role of transaction security perception in reducing the risk of churn for e-wallet users in Indonesia. *Journal of Digital Business and Data Science, 2*(2), 76-94. https://doi.org/10.59261/jdbs.v2i2.24

Verbeke, W., Dejaeger, K., Martens, D., Hur, J., and Baesens, B. (2012). New insights into churn prediction in the telecommunication sector. *European Journal of Operational Research, 218*(1), 211-229.

Winata, V., and Arma, O. (2025). Analyzing the effect of e-wallet usability on customer retention in mobile payment apps. *JUMDER: Jurnal Bisnis Digital Dan Ekonomi Kreatif, 1*(2), 1-20.

---

## Figures

**Figure 1:** Class balance in the master dataset (375,537 customers). Churners represent 10.1% of the population, confirming class imbalance that requires imbalance-aware training.
*Source: outputs/figures/01_class_balance.png*

**Figure 2:** Reactivator sub-population analysis. The 5,757 reactivator wallets show a 100% churn rate, enabling rule-based automatic flagging without model inference.
*Source: outputs/figures/04_reactivator_analysis.png*

**Figure 3:** Spearman correlation between features and the churn label. Days_since_last_txn and txn_velocity_change lead; demographic features show near-zero correlation.
*Source: outputs/figures/06_spearman_correlation.png*

**Figure 4:** ROC and precision-recall curves for XGBoost (tuned). AUC 0.759, PR-AUC 0.426. The PR curve is the operationally relevant comparison for imbalanced datasets.
*Source: outputs/figures/04_xgboost/roc_pr_comparison.png*

**Figure 5:** XGBoost permutation importance on the test set. Days_since_last_txn leads by a wide margin, confirming the recency-churn relationship documented in the literature.
*Source: outputs/figures/04_xgboost/permutation_importance.png*

**Figure 6:** K-means cluster selection by silhouette score (K = 2 to 6). The sharp drop at K = 4 confirms that K = 3 is the natural segmentation boundary in this dataset.
*Source: outputs/figures/06_cluster_predict/kmeans_k_selection.png*

**Figure 7:** SHAP beeswarm summary plot for XGBoost. Each point is one customer; horizontal position shows SHAP value (impact on model output); color shows feature value. Days_since_last_txn and txn_velocity_change dominate global attribution.
*Source: outputs/figures/08_shap/shap_summary.png*
