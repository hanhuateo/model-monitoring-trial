import pandas as pd
from model_monitoring import ModelMonitoring

train_df = pd.read_csv("../data/cleaned_employee_train.csv")
test_df = pd.read_csv("../data/cleaned_employee_test.csv")
model_monitoring = ModelMonitoring(train_df.drop(columns=['target', 'prediction']))
# print(model_monitoring.categorical_columns)
# print(model_monitoring.numerical_columns)
# Feature Drift
# model_monitoring.feature_drift_report(train_df=train_df, test_df=test_df, format='json')

# Target Drift
# model_monitoring.prediction_drift_report(train_df=train_df.drop(columns=['target']), test_df=test_df.drop(columns=['target']))
# print(RF_clf.feature_importances_)
# print(RF_clf.feature_names_in_)

# Data Quality
data_quality_dict = model_monitoring.data_quality_report(train_df=train_df.drop(columns=['target', 'prediction']), test_df=test_df, format='html')
