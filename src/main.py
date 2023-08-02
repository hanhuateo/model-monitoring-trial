import numpy as np
import pandas as pd
from joblib import load
from train_model import data_cleaning
from numpy import savetxt

"""
this part is to output the cleaned dataset for incoming data, which adds on the prediction column
into the employee_incoming.csv and output it as incoming_data_cleaned.csv
"""
RF_clf = load('./model/RF_clf.joblib')
column_transformer = load('./preprocessor/column_transformer.pkl')
label_encoder = load('./preprocessor/label_encoder.pkl')

incoming_df = pd.read_csv("../data/raw_split_data/employee_incoming.csv")
incoming_df = data_cleaning(incoming_df)
attrition = incoming_df['Attrition']

incoming_df = incoming_df.drop(columns=['Attrition'])
incoming_df_processed = column_transformer.transform(incoming_df)
incoming_df_processed = pd.DataFrame(incoming_df_processed, columns = [column_transformer.get_feature_names_out()])
incoming_df_processed.to_csv("../data/incoming_features_df_processed.csv", index=False)
attrition_pred = RF_clf.predict(incoming_df_processed)
attrition_pred_inverse = label_encoder.inverse_transform(attrition_pred)
incoming_df['prediction'] = attrition_pred_inverse
incoming_df['target'] = attrition
incoming_df.to_csv("../data/incoming_data_cleaned.csv", index=False)
"""
this part is to output the cleaned dataset for the test portion from the training dataset,
so that it can be used to compare drift with the incoming dataset
"""
test_features_df = pd.read_csv('../data/employee_features_test.csv')
test_ground_truth_df = pd.read_csv('../data/employee_ground_truth_test.csv')
test_features_df_processed = column_transformer.transform(test_features_df)
test_features_df_processed = pd.DataFrame(test_features_df_processed, columns = [column_transformer.get_feature_names_out()])
test_features_df_processed.to_csv("../data/test_features_df_processed.csv", index=False)
test_prediction = RF_clf.predict(test_features_df_processed)
test_prediction_inverse = label_encoder.inverse_transform(test_prediction)
test_features_df['prediction'] = test_prediction_inverse
test_features_df['target'] = test_ground_truth_df
test_features_df.to_csv("../data/test_data_cleaned.csv", index=False)