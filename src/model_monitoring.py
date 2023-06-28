import numpy as np
class ModelMonitoring():
    def __init__(self):
        self.numerical_columns = []
        self.categorical_columns = []
        self.stat_test_foreach_column = {}

    def get_numerical_columns(self, df):
        list_column_names = list(df.columns)
        numerical_list = []
        for col in list_column_names:
            if ((df[col].dtype in [np.int64,float]) and (df[col].nunique())) > 5:
                # print("Inside Numerical")
                numerical_list.append(col)
                continue
        return numerical_list

    def get_categorical_columns(self, df):
        list_column_names = list(df.columns)
        categorical_list = []
        for col in list_column_names:
            if col not in self.numerical_columns:
                categorical_list.append(col)
        return categorical_list

    def feature_drift(self, df):
        self.numerical_columns = self.get_numerical_columns(df)
        self.categorical_columns = self.get_categorical_columns(df)
        
