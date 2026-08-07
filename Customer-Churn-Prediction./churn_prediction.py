import pandas as pd 

df = pd.read_csv("dataset.csv")

print(df[df['TotalCharges'].str.strip() == ''][['customerID', 'tenure', 'TotalCharges']])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print(df['TotalCharges'].isnull().sum())
df = df.drop("customerID",axis =1 )
df['Churn'] = df['Churn'].map({'Yes':1, 'No': 0})

print(df.head())
print(df.info())

print(df['Churn'].mean())
print(df.groupby('Contract')['Churn'].mean())

import matplotlib.pyplot as plt

df.groupby('Contract')['Churn'].mean().plot(kind='bar')
plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print(df.groupby('InternetService')['Churn'].mean())
print(df.groupby('Churn')['tenure'].mean())

cat_cols = df.select_dtypes(include='object').columns.tolist()
print(cat_cols)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print(df.dtypes)

from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape, X_test.shape)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=1)
print("Without stratify:", y_test1.mean())

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
print("With stratify:", y_test2.mean())

print("Full dataset:", y.mean())

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("Mean of first column, unscaled:", X_train['gender'].mean())
print("Mean of first column, scaled:", X_train_scaled[:, 0].mean())
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
logreg.fit(X_train_scaled, y_train)

pred = logreg.predict(X_test_scaled)
accuracy = (pred == y_test).mean()
print("Accuracy:", accuracy)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
accuracy_rf = (pred_rf == y_test).mean()
print("Random Forest Accuracy:", accuracy_rf)

from xgboost import XGBClassifier

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                     scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

pred_xgb = xgb.predict(X_test)
accuracy_xgb = (pred_xgb == y_test).mean()
print("XGBoost Accuracy:", accuracy_xgb)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Logistic Regression
proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]
print("LogReg  - Precision:", precision_score(y_test, pred), "Recall:", recall_score(y_test, pred),
      "F1:", f1_score(y_test, pred), "ROC-AUC:", roc_auc_score(y_test, proba_lr))

# Random Forest
proba_rf = rf.predict_proba(X_test)[:, 1]
print("RF      - Precision:", precision_score(y_test, pred_rf), "Recall:", recall_score(y_test, pred_rf),
      "F1:", f1_score(y_test, pred_rf), "ROC-AUC:", roc_auc_score(y_test, proba_rf))

# XGBoost
proba_xgb = xgb.predict_proba(X_test)[:, 1]
print("XGBoost - Precision:", precision_score(y_test, pred_xgb), "Recall:", recall_score(y_test, pred_xgb),
      "F1:", f1_score(y_test, pred_xgb), "ROC-AUC:", roc_auc_score(y_test, proba_xgb))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
print(cm)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(10))