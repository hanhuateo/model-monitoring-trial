import pandas as pd
import numpy as np
from joblib import load
from model_monitoring import ModelMonitoring

RF_clf = load('./model/RF_clf.joblib')
column_transformer = load('./preprocessor/column_transformer.pkl')
label_encoder = load("./preprocessor/label_encoder.pkl")

model_monitoring = ModelMonitoring()
test_df = pd.read_csv("../data/raw_split_data/employee_test.csv")
train = pd.read_csv("../data/train_data/employee_train_features.csv")

# data cleaning
test_df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], inplace=True)
test_df = test_df.replace({'BusinessTravel': {'Non-Travel':1,
                                    'Travel_Rarely':2,
                                    'Travel_Frequently':3}
                 })

# data understanding
feature_names = test_df.drop(columns='Attrition').columns.to_list()
print(f"feature names are : {feature_names}")

# Categorical features
# Nominal
nominal_features = test_df.drop(columns=['Attrition']).select_dtypes(include=['object']).columns.tolist()
print(f"Nominal features are : {nominal_features}")
test_df[nominal_features] = test_df[nominal_features].astype('category')

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

# Data Preprocessing
X = test_df.drop(columns=['Attrition'])
X.to_csv("../data/test_data/employee_test_features.csv", index=False)
y = test_df['Attrition']
y.to_csv("../data/test_data/employee_test_attrition_ground_truth.csv", index=False)

# Feature Drift
model_monitoring.feature_drift_report(train_df=train, test_df=X)

X_processed = column_transformer.transform(X)
y_pred = RF_clf.predict(X_processed)

