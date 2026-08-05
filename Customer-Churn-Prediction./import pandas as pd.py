import pandas as pd 

df = pd.read_csv("dataset.csv")

# df.head()
# df.info()

# print(df[df['TotalCharges'].str.strip() == ''])

print(df[df['TotalCharges'].str.strip() == ''][['customerID', 'tenure', 'TotalCharges']])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print(df['TotalCharges'].isnull().sum())

df = df.drop('customerID', axis=1)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})