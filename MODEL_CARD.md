# Model Card: Tuned XGBoost Credit-Default Benchmark

## Model details

| Field | Value |
|---|---|
| Model name | Tuned XGBoost Full Benchmark |
| Version | 1.0 |
| Status | Portfolio / research demonstration only |
| Target | Default in the following month |
| Output | Predicted probability of default |
| Validated threshold | 0.19 under an illustrative 5:1 FN-to-FP cost ratio |
| Primary implementation | scikit-learn Pipeline with XGBoost classifier |
| Last validation date | 2026-10-12 |

## Intended use

This model demonstrates an end-to-end credit-risk model-development and
validation workflow. Appropriate uses include:

- portfolio demonstration for credit-risk, fraud/risk analytics, model-risk,
  decision-science, and data-analytics roles;
- education on cross-validation, threshold selection, calibration, SHAP, and
  subgroup diagnostics;
- reproducible experimentation on a public benchmark dataset.

## Out-of-scope and prohibited use

The model is not approved for:

- real credit approval, denial, limit assignment, pricing, or collections;
- adverse-action notices or explanations;
- individual financial advice;
- production decisions without new data, legal review, independent validation,
  monitoring, documentation, and appropriate human governance.

## Data

- **Source:** UCI Machine Learning Repository dataset 350, *Default of Credit
  Card Clients*.
- **Population:** 30,000 Taiwanese credit-card customers.
- **Reference period:** 2005.
- **Target prevalence:** 22.12% default and 77.88% non-default.
- **Split:** stratified 80% development and 20% holdout using
  `random_state=42`.
- **Holdout size:** 6,000 customers, including 1,327 defaults.

The historical population, country, product, and economic environment limit
generalizability to modern portfolios.

## Inputs and feature engineering

The model uses credit limit, age, repayment-status history, bill amounts,
payment amounts, and demographic fields from the public dataset. Five derived
features summarize:

1. number of delinquent months;
2. maximum delinquency severity;
3. latest credit utilization;
4. average six-month utilization;
5. maximum six-month utilization.

Sex, education, and marital status are included in the full benchmark. A
restricted challenger excludes these three variables.

## Development and validation procedure

1. Rule-based category cleaning and numerical plausibility checks.
2. Stratified development–holdout split before exploratory analysis.
3. Five-fold stratified development cross-validation.
4. Logistic-regression baseline with robust scaling and one-hot encoding.
5. XGBoost challenger with one-hot encoding and sparse preprocessing.
6. Randomized hyperparameter search optimized for average precision.
7. Cost-sensitive threshold selection from development out-of-fold
   probabilities.
8. One-time final assessment on the untouched holdout sample.
9. Calibration, SHAP, sensitive-feature, and subgroup analyses.

Hyperparameter selection and the tuned cross-validation summary use the same
development sample. Those values may be mildly optimistic and are treated as
development estimates. Holdout metrics are the final reported benchmark.

## Performance

### Development cross-validation

| Model | ROC-AUC | PR-AUC | Balanced accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.7726 | 0.5430 | 0.6562 | 0.6827 | 0.3601 |
| Baseline XGBoost | 0.7866 | 0.5606 | 0.6602 | 0.6739 | 0.3716 |
| Tuned XGBoost | 0.7878 | 0.5656 | 0.6608 | 0.6791 | 0.3714 |

### Final holdout ranking and calibration

| Metric | Result |
|---|---:|
| ROC-AUC | 0.7835 |
| PR-AUC | 0.5621 |
| KS statistic | 0.4341 |
| Brier score | 0.1341 |
| Binned expected calibration error | 0.0104 |
| Brier skill score | 0.2218 |

### Operating-point comparison

| Metric | Threshold 0.50 | Threshold 0.19 |
|---|---:|---:|
| Accuracy | 81.98% | 71.25% |
| Balanced accuracy | 65.72% | 70.24% |
| Precision | 66.99% | 41.01% |
| Recall | 36.55% | 68.43% |
| Specificity | 94.89% | 72.05% |
| F1 | 47.29% | 51.28% |
| High-risk rate | 12.07% | 36.90% |
| False positives | 239 | 1,306 |
| False negatives | 842 | 419 |
| Illustrative cost | 4,449 | 3,401 |

The 0.19 threshold reduced the illustrative holdout cost by 23.6%. It is not a
universal cutoff: a real institution must estimate costs from exposure, loss
given default, recovery, operational capacity, and policy constraints.

## Explainability

Global SHAP analysis identified delinquency severity and persistence as the
largest predictive contributors. Latest repayment status, utilization, credit
limit, bill amounts, and payment amounts also contributed.

SHAP values are calculated on the same sparse representation used for model
scoring. A prediction-fidelity assertion verifies a maximum difference of zero
between Pipeline predictions and direct transformed-input predictions before
SHAP is computed.

SHAP values describe model behaviour, not causality. Rare categories may have
large local effects while contributing little to population-level importance.

## Demographic-feature and subgroup review

Removing sex, education, and marital status changed development performance by:

| Metric | Difference without demographics |
|---|---:|
| ROC-AUC | -0.0013 |
| PR-AUC | -0.0027 |
| Balanced accuracy | +0.0003 |
| Precision | -0.0012 |
| Recall | +0.0009 |

The restricted challenger is preferable for future governance consideration,
but it has not been evaluated on a new independent holdout. Reported holdout
metrics belong to the full benchmark.

Subgroup metrics are descriptive. The largest populated groups showed broadly
similar precision, with some variation in true-positive and false-positive
rates. The Other/Unknown education and marital-status groups were too small for
reliable conclusions. This audit is not a legal-fairness determination.

## Risks and limitations

- **Temporal relevance:** the data are from 2005.
- **Population shift:** results may not transfer across markets or products.
- **Random split:** no out-of-time validation is available.
- **Selection uncertainty:** tuned development metrics are not nested-CV
  estimates.
- **Cost uncertainty:** the 5:1 ratio is illustrative.
- **Sensitive attributes:** the full model contains demographic variables.
- **Small groups:** subgroup estimates can be unstable.
- **Causal misuse:** predictive associations must not be treated as causal.
- **Operational risk:** no production data-quality, latency, security, override,
  or adverse-action process is implemented.

## Required controls before any production consideration

1. Retrain and validate on current, representative, legally permissible data.
2. Conduct independent out-of-time and external validation.
3. Validate a model excluding protected or sensitive attributes.
4. Establish policy-approved cost estimates and capacity constraints.
5. Perform legal, compliance, privacy, and fair-lending review.
6. Define performance, calibration, drift, and subgroup monitoring thresholds.
7. Implement challenger governance, overrides, incident handling, and periodic
   revalidation.
8. Create decision-specific, legally appropriate explanations.

## Reproducibility artifacts

- `credit_risk_model_validation.ipynb`: complete development and validation.
- `models/tuned_xgboost_full_model.joblib`: fitted full benchmark pipeline.
- `models/model_metadata.json`: threshold, metrics, features, and versions.
- `reports/tables/`: ranking, threshold, calibration, SHAP, and subgroup tables.
- `figures/`: calibration and SHAP figures.
- `app.py`: interactive validation dashboard.

## Approval statement

This model has no production approval. It is a transparent portfolio artifact
designed to demonstrate sound analytical reasoning and model-risk awareness.
