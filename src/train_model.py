import pandas as pd
import numpy as np
# from numpy import savetxt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from joblib import dump

# this part is to split the original dataset into a reference dataset (X_train) and an unseen dataset (X_test)
# will not be used again after first time running

# df = pd.read_csv("../data/raw/employee.csv")
# X_train, X_test = train_test_split(df, test_size=0.5, random_state=42, stratify=df['Attrition'])
# print(X_train)
# print(X_test)
# X_train.reset_index(drop=True, inplace=True)
# X_test.reset_index(drop=True, inplace=True)
# print(X_train)
# print(X_test)
# X_train.to_csv("../data/raw_split_data/employee_train.csv", index=False)
# X_test.to_csv("../data/raw_split_data/employee_test.csv", index=False)

# this part is to go through the standard data science pipeline to train the model

# data cleaning
def data_cleaning(df):
    df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], inplace=True)
    df = df.replace({'BusinessTravel': {'Non-Travel':1,
                                        'Travel_Rarely':2,
                                        'Travel_Frequently':3}
                    })
    return df

# data understanding
def data_understanding(df):
    feature_names = df.drop(columns='Attrition').columns.to_list()
    print(f"feature names are : {feature_names}")
    # feature names are : ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

    # Categorical features
    # Nominal
    nominal_features = df.drop(columns=['Attrition']).select_dtypes(include=['object']).columns.tolist()
    print(f"Nominal features are : {nominal_features}")
    df[nominal_features] = df[nominal_features].astype('category')
    # Nominal features are : ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

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
    # Ordinal features are : ['BusinessTravel', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']

    # Numerical features
    numerical_features = [feature for feature in feature_names if feature not in nominal_features + ordinal_features]
    print(f"Numerical features are : {numerical_features}")
    # Numerical features are : ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Skipped EDA

# Data Preprocessing
def data_preprocessing(df):
    X = df.drop(columns=['Attrition'])
    # X.to_csv("../data/train_data/employee_train_features.csv", index=False)
    y = df['Attrition']
    # y.to_csv("../data/train_data/employee_attrition_ground_truth.csv", index=False)

    columns_list = X.columns.tolist()
    dump(columns_list, './preprocessor/training_column_names_list.joblib')

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
    
    # X_train_processed, y_train_processed, X_test_processed, y_test_processed = X_train, y_train, X_test, y_test
    return X_train_processed, y_train_processed, X_test_processed, y_test_processed

# random forest
def RandomForestModel(X_train_processed, y_train_processed, X_test_processed, y_test_processed):
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
    # print(f"X_train_processed : {X_train_processed}")
    # print(f"y_train_processed : {y_train_processed}")
    RF_clf = RandomForestClassifier(**RF_search.best_params_, class_weight='balanced', random_state=42)
    RF_clf.fit(X_train_processed, y_train_processed)
    dump(RF_clf, './model/RF_clf.joblib')

    y_train_pred = RF_clf.predict(X_train_processed)
    y_train_pred_prob = RF_clf.predict_proba(X_train_processed)[:, 1]
    train_accuracy = accuracy_score(y_train_processed, y_train_pred)
    train_f1  = f1_score(y_train_processed, y_train_pred)
    train_roc_auc = roc_auc_score(y_train_processed, y_train_pred_prob)
    train_averaege_precision = average_precision_score(y_train_processed, y_train_pred_prob)
    print(f"Train metrics are: accuracy = {train_accuracy}, f1 = {train_f1}, roc_auc = {train_roc_auc}, average precision = {train_averaege_precision}")
    # Train metrics are: accuracy = 0.985480943738657, f1 = 0.9545454545454546, roc_auc = 0.9964902807775378, average precision = 0.9781413636799903

    y_test_pred = RF_clf.predict(X_test_processed)
    y_test_pred_prob = RF_clf.predict_proba(X_test_processed)[: ,1]
    test_accuracy = accuracy_score(y_test_processed, y_test_pred)
    test_f1 = f1_score(y_test_processed, y_test_pred)
    test_roc_auc = roc_auc_score(y_test_processed, y_test_pred_prob)
    test_average_precision = average_precision_score(y_test_processed, y_test_pred_prob)
    print(f"Test metrics are: accuracy = {test_accuracy}, f1 = {test_f1}, roc_auc = {test_roc_auc}, average precision = {test_average_precision}")
    # Test metrics are: accuracy = 0.8641304347826086, f1 = 0.4897959183673469, roc_auc = 0.7891774891774891, average precision = 0.5623943331157689

def main():
    # read dataset
    df = pd.read_csv("../data/raw_split_data/employee_train.csv")
    df = data_cleaning(df)
    data_understanding(df)
    X_train_processed, y_train_processed, X_test_processed, y_test_processed = data_preprocessing(df)
    RandomForestModel(X_train_processed, y_train_processed, X_test_processed, y_test_processed)

if __name__ == "__main__":
    main()
    
