import pandas as pd
from data import cleaning, preprocessing

df = pd.read_csv("./data/raw/employee.csv")

df = cleaning.drop_columns(df)

df = cleaning.ordinal_encoding(df)

