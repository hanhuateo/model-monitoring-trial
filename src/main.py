import numpy as pandas
import seaborn as sns
import pandas as pd
from joblib import load
from monitoring.evidently import *
from data_manipulation.cleaning import *
from data_manipulation.understandings import *
from data_manipulation.preprocess import *
from joblib import load
# metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve

RF_clf = load('../data/models/RF_clf.joblib')
reference = pd.read_csv('../data/split_data/train_dataset.csv')
current = pd.read_csv('../data/split_data/test_dataset.csv')
current = object_to_category(current)
current = drop_columns(current)
current = ordinal_encoding(current)
current.drop(columns='Attrition', inplace=True)
column_transformer = load('../data/preprocessor/column_transformer.pkl')
X_processed = test_preprocessing(current, column_transformer)
y_prediction = RF_clf.predict(X_processed)
y_pred_prob = RF_clf.predict_proba(X_processed)[:,1]
y_true = pd.read_csv('../data/split_data/y_combined.csv')
accuracy = accuracy_score(y_true, y_prediction)
f1 = f1_score(y_true, y_prediction)
roc_auc = roc_auc_score(y_true, y_pred_prob)
average_precision = average_precision_score(y_true, y_pred_prob)
print('accuracy : {}'.format(accuracy))
print('f1 : {}'.format(f1))
print('roc_auc : {}'.format(roc_auc))
print('average precision score : {}'.foramt(average_precision))