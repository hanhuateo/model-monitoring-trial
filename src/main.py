import pandas as pd
from numpy import savetxt
from joblib import load
from model_monitoring import ModelMonitoring

RF_clf = load('./model/RF_clf.joblib')
column_transformer = load('./preprocessor/column_transformer.pkl')
label_encoder = load("./preprocessor/label_encoder.pkl")

model_monitoring = ModelMonitoring()

test_df = pd.read_csv("../data/raw_split_data/employee_test.csv")
train_df = pd.read_csv("../data/raw_split_data/employee_train.csv")

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
X_test = test_df.drop(columns=['Attrition'])
X_test.to_csv("../data/test_data/employee_test_features.csv", index=False)
y_test = test_df['Attrition']
y_test.to_csv("../data/test_data/employee_test_attrition_ground_truth.csv", index=False)
# Feature Drift
model_monitoring.feature_drift_report(train_df=train_df.drop(columns=['Attrition']), test_df=X_test)
# the line of code above produces a warning that looks something like
# WARNING:root:Column Gender have different types in reference object and current category. Returning type from reference
# this warning stems from the fact that in line 30, we have set the dtype of object to category
# even though we have also done so in train_model.py, 
# this is most likely undone when we read employee_train_features.csv

X_test_processed = column_transformer.transform(X_test)
y_test_pred = RF_clf.predict(X_test_processed)
y_test_pred_inverse = label_encoder.inverse_transform(y_test_pred)
test_df['prediction'] = y_test_pred_inverse

# X_train_processed = column_transformer.