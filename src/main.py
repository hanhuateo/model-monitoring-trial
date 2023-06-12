import numpy as pandas
import seaborn as sns
import pandas as pd
from joblib import load
from monitoring.evidently import *
from data_manipulation.cleaning import *
from data_manipulation.understandings import *
from data_manipulation.preprocess import *
from joblib import load
from numpy import savetxt
# metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve

# load model and load preprocessing objects
RF_clf = load('../data/models/RF_clf.joblib')
column_transformer = load('../data/preprocessor/column_transformer.pkl')
label_encoder = load('../data/preprocessor/label_encoder.pkl')

# load reference and current data set
# reference = pd.read_csv('../data/split_data/train_dataset.csv')
current = pd.read_csv('../data/split_data/test_dataset.csv')

# data cleaning for current dataset
current = object_to_category(current)
current = drop_columns(current)
current = ordinal_encoding(current)

# obtain ground truth
y_true = current['Attrition']
print("y_true is : {}".format(y_true))
print(f"Shape of y_true is {y_true.shape}")
current.drop(columns='Attrition', inplace=True)
print(current.head())
current.to_csv('../data/split_data/X_current.csv')
y_true.to_csv('../data/split_data/y_current.csv')

# do preprocessing on features
X_processed = test_preprocessing(current, column_transformer)
print("X_processed is : {}".format(X_processed))

# obtain the predictions
y_prediction = RF_clf.predict(X_processed)
print("y_prediction is : {}".format(y_prediction))
print(f"Shape of y_prediction is {y_prediction.shape}")

y_pred_prob = RF_clf.predict_proba(X_processed)[:,1]

# do preprocessing on label
y_true_processed = y_preprocessing(y_true, label_encoder)

accuracy = accuracy_score(y_true_processed, y_prediction)
f1 = f1_score(y_true_processed, y_prediction)
roc_auc = roc_auc_score(y_true_processed, y_pred_prob)
average_precision = average_precision_score(y_true_processed, y_pred_prob)
print('accuracy : {}'.format(accuracy))
print('f1 : {}'.format(f1))
print('roc_auc : {}'.format(roc_auc))
print('average precision score : {}'.format(average_precision))