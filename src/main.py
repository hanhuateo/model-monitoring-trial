import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve

from joblib import dump, load

# this part is to split the original dataset into a reference dataset (X_train) and an unseen dataset (X_test)
"""
df = pd.read_csv("../data/raw/employee.csv")
X_train, X_test = train_test_split(df, test_size=0.5, random_state=42)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
X_train.to_csv("../data/split_data/employee_train.csv")
X_test.to_csv("../data/split_data/employee.test.csv")
"""

# this part is to go through the standard data science pipeline to train the model
# and also to fit in the evidently scripts that has been written for 
# feature drift and prediction drift

# read dataset
df = pd.read_csv("../data/split_data/employee_train.csv")

# data cleaning
df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], inplace=True)
df = df.replace({'BusinessTravel': {'Non-Travel':1,
                                    'Travel_Rarely':2,
                                    'Travel_Frequently':3}
                 })

# data understanding
feature_names = df.drop(columns='Attrition').columns.to_list()
print(f"feature names are : {feature_names}")

# Categorical features
# Nominal
nominal_features = df.drop(columns=['Attrition']).select_dtypes(include=['object']).columns.tolist()
print(f"Nominal features are : {nominal_features}")
df[nominal_features] = df[nominal_features].astype('category')

# Ordinal
ordinal_features_mapping = {'BusinessTravel': {1: 'Non-Travel', 2: 'Travel_Rarely', 3: 'Travel_Frequently'},
                            'Education': {1: 'Below College', 2: 'College', 3: 'Bachelor',  4: 'Master', 5: 'Doctor'},
                            'EnvironmentSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
                            'JobInvolvement': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
                            'JobLevel': {1: 'Junior', 2: 'Mid', 3: 'Senior', 4: 'Principal', 5: 'Head'},
                            'JobSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
                            'PerformanceRating': {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'},
                            'RelationshipSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
                            'StockOptionLevel': {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'},
                            'WorkLifeBalance': {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
                            }
ordinal_features = list(ordinal_features_mapping.keys())
print(f"Ordinal features are : {ordinal_features}")

# Numerical features
numerical_features = [feature for feature in feature_names if feature not in nominal_features + ordinal_features]
print(f"Numerical features are : {numerical_features}")

# Skipped EDA

# Data Preprocessing
X = df.drop(columns=['Attrition'])
y = df['Attrition']
y.to_csv("../data/employee_attrition_ground_truth.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# shape of train set, test set
print(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, Shape of y_test: {y_test.shape}")

print("Train set:")
print(round(y_train.value_counts(normalize=True), 4))
print()
print("Test set:")
print(round(y_test.value_counts(normalize=True), 4))

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# Preprocessing Pipeline
# define categorical and numerical transformers
categorical_transformer = Pipeline(steps=[
    # ('SimpleImputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop=None))
])

numerical_transformer = Pipeline(steps=[
    # ('knnImputer', KNNImputer(n_neighbors=3, weights="uniform")),
    ('scaler', StandardScaler())
])


#  dispatch object columns to the categorical_transformer and remaining columns to numerical_transformer
preprocessor = ColumnTransformer(transformers=[
    ('categorical', categorical_transformer, make_column_selector(dtype_include="category")),
    ('numerical', numerical_transformer, make_column_selector(dtype_exclude="category"))
])

# using make_column_transformer
X_train_processed = preprocessor.fit_transform(X_train)
print(f"Shape of X_train_processed after preprocessing: {X_train_processed.shape}")

X_test_processed = preprocessor.transform(X_test)
print(f"Shape of X_test_processed after preprocessing: {X_test_processed.shape}")

dump(preprocessor, './preprocessor/column_transformer.pkl')

LE = LabelEncoder()
y_train_processed = LE.fit_transform(y_train)
print(f"Shape of y_train_processed after preprocessing: {y_train_processed.shape}")

y_test_processed = LE.transform(y_test)
print(f"Shape of y_test_processed after preprocessing: {y_test_processed.shape}")

dump(LE, './preprocessor/label_encoder.pkl')

# random forest
RF_clf = RandomForestClassifier(class_weight='balanced', random_state=42)

param_grid = {'n_estimators': [100, 500, 900], 
              'max_features': ['auto', 'sqrt'],
              'max_depth': [2, 15, None], 
              'min_samples_split': [5, 10],
              'min_samples_leaf': [1, 4], 
              'bootstrap': [True, False]
              }

# hyperparameter tuning
RF_search = GridSearchCV(RF_clf, param_grid=param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
RF_search.fit(X_train_processed, y_train_processed)

RF_clf = RandomForestClassifier(**RF_search.best_params_, class_weight='balanced', random_state=42)
RF_clf.fit(X_train_processed, y_train_processed)

dump(RF_clf, './model/RF_clf.joblib')