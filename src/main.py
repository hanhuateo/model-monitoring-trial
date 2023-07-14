import pandas as pd
from joblib import load
from model_monitoring import ModelMonitoring
from train_model import data_cleaning
RF_clf = load('./model/RF_clf.joblib')
column_transformer = load('./preprocessor/column_transformer.pkl')
label_encoder = load("./preprocessor/label_encoder.pkl")

test_df = pd.read_csv("../data/raw_split_data/employee_test.csv")
# data cleaning
test_df = data_cleaning(test_df)
# Data Preprocessing
X_test = test_df.drop(columns=['Attrition'])
y_test = test_df['Attrition']
X_test_processed = column_transformer.transform(X_test)
y_test_pred = RF_clf.predict(X_test_processed)
y_test_pred_inverse = label_encoder.inverse_transform(y_test_pred)
y_test_pred_prob = RF_clf.predict_proba(X_test_processed)[:1]
print(f"y test prediction probability : {y_test_pred_prob}")
test_df['prediction'] = y_test_pred_inverse
test_df.rename(columns={'Attrition' : 'target'}, inplace=True)

train_df = pd.read_csv("../data/cleaned_employee_train.csv")

model_monitoring = ModelMonitoring(train_df)
print(model_monitoring.categorical_columns)
# Feature Drift
# model_monitoring.feature_drift_report(train_df=train_df, test_df=test_df, format='json')

# Target Drift
# model_monitoring.prediction_drift_report(train_df=train_df.drop(columns=['target']), test_df=test_df.drop(columns=['target']))
# print(RF_clf.feature_importances_)
# print(RF_clf.feature_names_in_)