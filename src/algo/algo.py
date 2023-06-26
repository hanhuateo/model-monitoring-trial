import pandas as pd
from scipy import stats
def data_type_algo(df):
    list_column_names = list(df.columns)
    numerical_list = []
    nominal_list = []
    ordinal_list = []
    for col in list_column_names:
        # first if statement to determine ordinal column type
        if df[col].dtype in [int,float] and df[col].nunique() <= 5:
            ordinal_list.append(col)
            continue
        if df[col].dtype in [int,float] and df[col].nunique() > 5:
            numerical_list.append(col)
            continue
        if df[col].dtype in [object, str, 'category']:
            nominal_list.append(col)
            continue
    return numerical_list, nominal_list, ordinal_list
    
def categorical_stat_test_algo(df, data_type_list, num_of_rows, per_column_dictionary):
    if (num_of_rows <= 1000):
        for col in data_type_list:
            if df[col].nunique > 2:
                per_column_dictionary.update({col: 'chisquare'})
            else:
                per_column_dictionary.update({col: 'z'})
    else:
        for col in data_type_list:
            per_column_dictionary.update({col: 'jensenshannon'}) # can be JS or KL 
    
    return per_column_dictionary

def numerical_stat_test_algo(df, data_type_list, num_of_rows, per_column_dictionary):
    if (num_of_rows <= 1000):
        for col in data_type_list:
            per_column_dictionary.update({col:'ks'})
    else:
        for col in data_type_list:
            if (num_of_rows < 5000):
                res = stats.shapiro(df[col])
            else:
                res = stats.anderson(df[col])
            if (res.statistic > 0.7):
                per_column_dictionary.update({col:'t_test'})
            else:
                per_column_dictionary.update({col:'wasserstein'}) # can be cramer, mann-whitney, wasser, JS, KL 
