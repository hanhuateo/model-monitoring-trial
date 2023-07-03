import pandas as pd
import numpy as np
from joblib import load
from model_monitoring import ModelMonitoring

RF_clf = load('./model/RF_clf.joblib')
column_transformer = load('./preprocessor/column_transformer.pkl')
label_encoder = load("./preprocessor/label_encoder.pkl")

model_monitoring = ModelMonitoring()
test_df = pd.read_csv("../data/split_data/employee_test.csv")
train_df = pd.read_csv("../data/split_data/employee_train.csv")
ModelMonitoring.feature_drift_report(model_monitoring, train_df=train_df, test_df=test_df)