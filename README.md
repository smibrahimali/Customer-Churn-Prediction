# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their service, using the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features).

## Problem

Customer acquisition is expensive, so identifying at-risk customers *before* they leave lets a business intervene (discounts, outreach, support) instead of losing them outright. This project builds and compares several models that flag customers likely to churn, and explains what's actually driving that risk.

## Approach

1. **Cleaning** — `TotalCharges` was loaded as text instead of a number. Investigating why revealed it wasn't random: every affected row belonged to a customer with `tenure == 0` (brand-new signups with no bill yet), so those values were filled with `0` rather than the column average.
2. **EDA** — explored churn patterns by contract type, internet service, and tenure before touching any model.
3. **Preprocessing** — label-encoded categorical features, scaled numeric features for the linear model, and used a stratified train/test split (80/20) to keep the true churn ratio consistent between train and test.
4. **Modeling** — trained and compared Logistic Regression (baseline), Random Forest, and XGBoost, each with class-imbalance handling (`class_weight='balanced'` / `scale_pos_weight`), since only ~26.5% of customers in the dataset actually churn.
5. **Evaluation** — went beyond accuracy to precision, recall, F1, and ROC-AUC, and reasoned about the *business* cost of each type of mistake before picking a final model.
6. **Explainability** — extracted feature importances and checked that they agreed with the EDA findings.

## Key findings

- **Contract type is the strongest churn driver.** Month-to-month customers churn at **42.7%**, vs **11.3%** for one-year and just **2.8%** for two-year contracts.
- **Churn happens early.** Customers who churned had an average tenure of **18.0 months**, roughly half that of customers who stayed (**37.6 months**).
- **Fiber optic customers churn the most by service type** — **41.9%**, compared to 19.0% for DSL and 7.4% for customers with no internet service.
- **Feature importance confirms the EDA**: `Contract` and `tenure` are the top two predictors in the Random Forest model, matching the patterns found by hand. `OnlineSecurity` and `TechSupport` also rank surprisingly high — customers without these add-ons appear notably more likely to churn.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.739 | 0.505 | **0.797** | 0.618 | 0.840 |
| Random Forest | **0.767** | **0.543** | 0.770 | **0.637** | **0.842** |
| XGBoost | 0.759 | 0.532 | 0.786 | 0.634 | 0.841 |

**Model selected: Logistic Regression**, despite not having the top accuracy. Reasoning: in this problem, a missed churner (false negative) costs the business far more than a false alarm (false positive) — a missed churner represents fully lost future revenue, while a false alarm costs only a retention offer or a support call. Logistic Regression has the highest recall of the three (79.7%), catching the most true churners. Its confusion matrix on the test set:

```
[[743  292]     TN=743  FP=292
 [ 76  298]]    FN=76   TP=298
```

Out of 374 actual churners in the test set, the model correctly identified 298 and missed 76.

**Note:** Logistic Regression initially threw a `ConvergenceWarning` when trained on unscaled features. Applying `StandardScaler` resolved the warning and produced a more reliably converged solution; predictions were unaffected in this case, but scaling is standard practice for numerical stability.

## Top 10 features by importance (Random Forest)

| Feature | Importance |
|---|---|
| Contract | 0.2095 |
| tenure | 0.1435 |
| TotalCharges | 0.1146 |
| MonthlyCharges | 0.1055 |
| OnlineSecurity | 0.1015 |
| TechSupport | 0.0773 |
| InternetService | 0.0594 |
| PaymentMethod | 0.0387 |
| OnlineBackup | 0.0309 |
| PaperlessBilling | 0.0203 |

## Limitations

- Categorical features were label-encoded rather than one-hot encoded, which imposes an artificial numeric order on features like `InternetService` (e.g., "Fiber optic" isn't inherently "between" "DSL" and "No"). This has little effect on tree-based models but can distort a linear model like Logistic Regression. One-hot encoding would be a natural next improvement.
- The dataset is a single snapshot per customer; it doesn't capture how usage or service changes over time.

## Project structure

```
├── dataset.csv
├── churn_prediction.py
└── README.md
```

## Running it

```bash
pip install pandas scikit-learn matplotlib seaborn xgboost
python churn_prediction.py
```

## Next steps

- Try one-hot encoding for categorical features and compare Logistic Regression's performance
- Hyperparameter tuning with `GridSearchCV`
- SHAP values for per-customer explanations
- A small Streamlit app for live churn predictions
