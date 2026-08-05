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
plt.show()