import pandas as pd
from data import cleaning, preprocess
from models import randomforest
from joblib import load

# read data
df = pd.read_csv("./data/split_data/train_dataset.csv")

# clean data
df = cleaning.drop_columns(df)
df = cleaning.ordinal_encoding(df)

# preprocessing
X_train, X_test, y_train, y_test = preprocess.split(df)
X_train_processed, y_train_processed, X_test_processed, y_test_processed = preprocess.preprocessing(X_train, X_test, y_train, y_test)

# train the model
randomforest.randomforestmodel(X_train_processed, y_train_processed)
RF_clf = load('./data/models/RF_clf.joblib')
