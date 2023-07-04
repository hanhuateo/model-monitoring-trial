import pandas as pd
import numpy as np
from joblib import load
df = pd.read_csv('../data/raw/employee.csv')
print(f"Shape of the dataset : {df.shape}")
print(f"Total Number of Columns are : {len(df.columns)}")
print(f"The columns are : {df.columns.tolist}")

RF_clf = load('./model/RF_clf.joblib')
print(f"Feature Importance : {RF_clf.feature_importances_}")