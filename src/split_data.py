import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/raw/employee.csv")
X_train, X_test = train_test_split(df, test_size=0.5, random_state=42, stratify=df['Attrition'])
print(X_train)
print(X_test)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
print(X_train)
print(X_test)
X_train.to_csv("../data/raw_split_data/employee_train.csv", index=False)
X_test.to_csv("../data/raw_split_data/employee_incoming.csv", index=False)