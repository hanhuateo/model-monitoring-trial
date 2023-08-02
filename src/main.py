import pandas as pd
from model_monitoring import ModelMonitoring

train_df = pd.read_csv("../data/cleaned_employee_train.csv")
test_df = pd.read_csv("../data/cleaned_employee_test.csv")
processed_train_df = pd.read_csv("../data/X_train_processed.csv")
processed_test_df = pd.read_csv("../data/X_test_processed.csv")
model_monitoring = ModelMonitoring()

# Feature Drift
# model_monitoring.feature_drift_report(train_df=train_df, test_df=test_df, format='html')

# Target Drift
# model_monitoring.prediction_drift_report(train_df=train_df.drop(columns=['target']), test_df=test_df.drop(columns=['target']))