import pandas as pd
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
    
