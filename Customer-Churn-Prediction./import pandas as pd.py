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