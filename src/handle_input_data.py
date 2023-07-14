import pandas as pd
from train_model import data_cleaning
from processed_feature_mapping import mapping
from joblib import load

RF_clf = load("./model/RF_clf.joblib")
label_encoder = load("./preprocessor/label_encoder.pkl")
column_transformer = load("./preprocessor/column_transformer.pkl")
test_df = pd.read_csv("../data/raw_split_data/employee_test.csv")
# data cleaning
test_df = data_cleaning(test_df)
# Data Preprocessing
X_test = test_df.drop(columns=['Attrition'])
X_test.to_csv('../data/cleaned_employee_test.csv', index=False)
y_test = test_df['Attrition']
X_test_processed = column_transformer.transform(X_test)
X_test_processed = pd.DataFrame.from_records(X_test_processed)
X_test_processed = mapping(X_test_processed, column_transformer)
X_test_processed.to_csv('../data/X_test_processed.csv', index=False)
y_test_pred = RF_clf.predict(X_test_processed)
y_test_pred_inverse = label_encoder.inverse_transform(y_test_pred)
y_test_pred_prob = RF_clf.predict_proba(X_test_processed)[:1]
print(f"y test prediction probability : {y_test_pred_prob}")
test_df['prediction'] = y_test_pred_inverse
test_df.rename(columns={'Attrition' : 'target'}, inplace=True)