# pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# preprocessing
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from scipy.sparse import save_npz

from numpy import savetxt, save

import pandas as pd

from joblib import dump

import csv

def split(df):
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # shape of train set, test set
    print(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, Shape of y_test: {y_test.shape}")

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    print("X_train is : {}".format(X_train)) 
    print("X_train type is : {}".format(type(X_train))) # dataframe
    print("X_test is : {}, {}".format(X_test, type(X_test))) # dataframe
    print("y_train is : {}, {}".format(y_train, type(y_train))) # series
    print("y_test is : {}, {}".format(y_test, type(y_test))) # series

    X_combined = pd.concat([X_train, X_test], axis=0)
    X_combined.to_csv('../data/split_data/X_combined.csv', index=False)

    y_combined = pd.concat([y_train, y_test], axis=0)
    y_combined.to_csv('../data/split_data/y_combined.csv', index=False)
    
    return X_train, X_test, y_train, y_test



def train_preprocessing(X_train, X_test, y_train, y_test):

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
    print('type of X_train_processed : {}'.format(type(X_train_processed).__name__))
    savetxt('../data/processed/X_train_processed.csv', X_train_processed, delimiter=',')

    X_test_processed = preprocessor.transform(X_test)
    print(f"Shape of X_test_processed after preprocessing: {X_test_processed.shape}")
    print('type of X_test_processed : {}'.format(type(X_test_processed).__name__))
    savetxt('../data/processed/X_test_processed.csv', X_test_processed, delimiter=',')

    df1 = pd.read_csv('../data/processed/X_train_processed.csv')
    df2 = pd.read_csv('../data/processed/X_test_processed.csv')
    X_combined_processed = pd.concat([df1, df2], axis=0)
    X_combined_processed.to_csv('../data/processed/X_combined_processed.csv', index=False)

    dump(preprocessor, '../data/preprocessor/column_transformer.pkl')

    LE = LabelEncoder()
    y_train_processed = LE.fit_transform(y_train)
    print(f"Shape of y_train_processed after preprocessing: {y_train_processed.shape}")
    savetxt('../data/processed/y_train_processed.csv', y_train_processed, delimiter=',')

    y_test_processed = LE.transform(y_test)
    print(f"Shape of y_test_processed after preprocessing: {y_test_processed.shape}")
    savetxt('../data/processed/y_test_processed.csv', y_test_processed, delimiter=',')

    df3 = pd.read_csv('../data/processed/y_train_processed.csv')
    df4 = pd.read_csv('../data/processed/y_test_processed.csv')
    y_combined_processed = pd.concat([df3, df4], axis=0)
    y_combined_processed.to_csv('../data/processed/y_combined_processed.csv', index=False)

    dump(LE, '../data/preprocessor/label_encoder.pkl')

    print(f"y_train after preprocessing: {y_train_processed}")
    print(f"y_train before preprocessing: {LE.inverse_transform(y_train_processed)}") 

    return X_train_processed, X_test_processed, y_train_processed, y_test_processed

def test_preprocessing(X, preprocessor):
    X_processed = preprocessor.transform(X)
    print(f"Shape of X after preprocessing: {X_processed.shape}")
    print('type of X_processed: {}'.format(type(X_processed).__name__))
    if (type(X_processed).__name__ == 'ndarray'):
        savetxt('../data/processed/X_current_processed.csv', X_processed, delimiter=',')
    else:
        save_npz('../data/processed/X_current_processed.npz', X_processed)
    return X_processed

def y_preprocessing(y, label_encoder):
    y_processed = label_encoder.transform(y)
    print("Shape of y after preprocessing : {}".format(y_processed.shape))
    print('type of y_processed: {}'.format(type(y_processed).__name__))
    if (type(y_processed).__name__ == 'ndarray'):
        savetxt('../data/processed/y_current_processed.csv', y_processed, delimiter=',')
    else:
        save_npz('../data/processed/y_current_processed.csv', y_processed)
    return y_processed