import pandas as pd
# from data_manipulation import cleaning, preprocess, understandings
from data_manipulation.cleaning import *
from data_manipulation.understandings import *
from data_manipulation.preprocess import *
from models.randomforest import *

# read data
df = pd.read_csv("./data/split_data/train_dataset.csv")

# clean data
df = drop_columns(df)
df = ordinal_encoding(df)

# convert type from object to category for nominal features
df = object_to_category(df)

# preprocessing
X_train, X_test, y_train, y_test = split(df)
X_train_processed, y_train_processed, X_test_processed, y_test_processed = preprocessing(X_train, X_test, y_train, y_test)

# train the model
RF_clf = randomforestmodel(X_train_processed, y_train_processed)
performance_metrics(RF_clf, X_train_processed, y_train_processed, "train")
performance_metrics(RF_clf, X_test_processed, y_test_processed, "test")
plot_evaluation_curves(RF_clf, X_train_processed, y_train_processed, X_test_processed, y_test_processed)