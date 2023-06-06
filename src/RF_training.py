import pandas as pd
# from data_manipulation import cleaning, preprocess, understandings
from data_manipulation.cleaning import *
from data_manipulation.understandings import *
from data_manipulation.preprocess import *
from models.randomforest import *
import sys

# read data
df = pd.read_csv("../data/split_data/train_dataset.csv")

# clean data
df = drop_columns(df)
df = ordinal_encoding(df)

# convert type from object to category for nominal features
df = object_to_category(df)

# preprocessing
X_train, X_test, y_train, y_test = split(df)
# shape of train set, test set
print(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, Shape of y_test: {y_test.shape}")

X_train_processed, y_train_processed, X_test_processed, y_test_processed = preprocessing(X_train, X_test, y_train, y_test)
print(f"Shape of X_train_processed: {X_train_processed.shape}, Shape of y_train_processed: {y_train_processed.shape}")
print(f"Shape of X_test_processed: {X_test_processed.shape}, Shape of y_test_processed: {y_test_processed.shape}")

# train the model
RF_clf = randomforestmodel(X_train_processed, y_train_processed)
performance_metrics(RF_clf, X_train_processed, y_train_processed, "train")
performance_metrics(RF_clf, X_test_processed, y_test_processed, "test")
plot_evaluation_curves(RF_clf, X_train_processed, X_test_processed, y_train_processed, y_test_processed)