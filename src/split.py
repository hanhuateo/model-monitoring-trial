import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV dataset into a pandas DataFrame
df = pd.read_csv("./data/raw/employee.csv")

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Write the training and testing sets into separate CSV files
train_df.to_csv('./data/split_data/train_dataset.csv', index=False)
test_df.to_csv('./data/split_data/test_dataset.csv', index=False)