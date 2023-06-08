import numpy as pandas
import seaborn as sns
import pandas as pd
from joblib import load
from monitoring.evidently import *
from data_manipulation.cleaning import *
from data_manipulation.understandings import *
from data_manipulation.preprocess import *
from joblib import load

RF_clf = load('../data/models/RF_clf.joblib')
reference = pd.read_csv('../data/split_data/train_dataset.csv')
current = pd.read_csv('../data/split_data/test_dataset.csv').drop(columns='Attrition', inplace=True)

X_processed = test_preprocessing(current)
