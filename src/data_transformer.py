import pandas as pd
import numpy as np
from joblib import load
from train_model import data_cleaning

df = pd.read_csv('../data/raw_split_data/employee_train.csv')
column_transformer = load('./preprocessor/column_transformer.pkl')
label_encoder = load('./preprocessor/label_encoder.pkl')
RF_clf = load('./model/RF_clf.joblib')
df = data_cleaning(df)
X_train = df.drop(columns=['Attrition'])
y_train = df['Attrition']
X_train_processed = column_transformer.transform(X_train)
y_train_pred = RF_clf.predict(X_train_processed)
y_train_pred_inverse = label_encoder.inverse_transform(y_train_pred)
df['prediction'] = y_train_pred_inverse
df.rename(columns={'Attrition' : 'target'}, inplace=True)
df.to_csv('../data/cleaned_employee_train.csv', index=False)