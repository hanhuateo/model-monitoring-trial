# pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# preprocessing
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

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

    return X_train, X_test, y_train, y_test

def preprocessing(X_train, X_test, y_train, y_test):
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

    LE = LabelEncoder()
    y_train_processed = LE.fit_transform(y_train)
    print(f"Shape of y_train_processed after preprocessing: {y_train_processed.shape}")

    y_test_processed = LE.transform(y_test)
    print(f"Shape of y_test_processed after preprocessing: {y_test_processed.shape}")

    print(f"y_train after preprocessing: {y_train_processed}")
    print(f"y_train before preprocessing: {LE.inverse_transform(y_train_processed)}") 

    return X_train_processed, y_train_processed, X_test_processed, y_test_processed