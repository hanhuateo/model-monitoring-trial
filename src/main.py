import pandas as pd
from numpy import savetxt
from joblib import load
from model_monitoring import ModelMonitoring
from train_model import data_cleaning, data_understanding
from processed_feature_mapping import mapping
RF_clf = load('./model/RF_clf.joblib')
column_transformer = load('./preprocessor/column_transformer.pkl')
label_encoder = load("./preprocessor/label_encoder.pkl")

model_monitoring = ModelMonitoring()

test_df = pd.read_csv("../data/raw_split_data/employee_test.csv")
# data cleaning
test_df = data_cleaning(test_df)
# data understanding
data_understanding(test_df)
# Data Preprocessing
X_test = test_df.drop(columns=['Attrition'])
# X_test.to_csv("../data/test_data/employee_test_features.csv", index=False)
y_test = test_df['Attrition']
# y_test.to_csv("../data/test_data/employee_test_attrition_ground_truth.csv", index=False)
X_test_processed = column_transformer.transform(X_test)
y_test_pred = RF_clf.predict(X_test_processed)
y_test_pred_inverse = label_encoder.inverse_transform(y_test_pred)
test_df['prediction'] = y_test_pred_inverse
test_df.rename(columns={'Attrition' : 'target'}, inplace=True)
# print(f"test data set : {test_df}")
X_test_processed_df = pd.DataFrame.from_records(X_test_processed)
X_test_processed_df = mapping(X_test_processed_df, column_transformer)
# X_test_processed_df.to_csv('../data/X_test_processed.csv', index=False)
X_test_processed_df = pd.read_csv('../data/X_test_processed.csv')

train_df = pd.read_csv("../data/cleaned_employee_train.csv")

processed_train_df = pd.read_csv("../data/X_train_processed.csv")

# Feature Drift
# model_monitoring.feature_drift_report(train_df=train_df, test_df=test_df)
# the line of code above produces a warning that looks something like
# WARNING:root:Column Gender have different types in reference object and current category. Returning type from reference
# this warning stems from the fact that in line 30, we have set the dtype of object to category
# even though we have also done so in train_model.py, 
# this is most likely undone when we read employee_train_features.csv

# Processed Feature Drift
# model_monitoring.processed_feature_drift_report(train_df=processed_train_df, test_df=X_test_processed_df)

# Target Drift
model_monitoring.target_drift_report(train_df=train_df.drop(columns=['target']), test_df=test_df.drop(columns=['target']))