import pandas as pd
import numpy as np
df = pd.read_csv('../data/raw/employee.csv')
print(f"Shape of the dataset : {df.shape}")
print(f"Total Number of Columns are : {len(df.columns)}")
print(f"The columns are : {df.columns.tolist}")