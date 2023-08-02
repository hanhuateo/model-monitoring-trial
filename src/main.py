import numpy as np
import pandas as pd
from joblib import load
from train_model import data_cleaning

RF_clf = load('./model/RF_clf.joblib')
column_transformer = load('./preprocessor/column_transformer.pkl')
label_encoder = load('./preprocessor/label_encoder.pkl')

incoming_df = pd.read_csv("../data/raw_split_data/employee_incoming.csv")
incoming_df = data_cleaning(incoming_df)
attrition = incoming_df['Attrition']

incoming_df = incoming_df.drop(columns=['Attrition'])
incoming_df_processed = column_transformer.transform(incoming_df)
attrition_pred = RF_clf.predict(incoming_df_processed)
attrition_pred_inverse = label_encoder.inverse_transform(attrition_pred)
incoming_df['prediction'] = attrition_pred_inverse
incoming_df['target'] = attrition
incoming_df.to_csv("../data/incoming_data_cleaned.csv", index=False)