import pandas as pd

def summary(df):
    df_summary = pd.DataFrame(data=[df.dtypes, df.nunique(), df.apply(lambda col: col.unique())],
                          columns=df.columns,
                          index=['dtype', 'n_unique', 'unique_values'])

    print(df_summary.transpose())

def drop_columns(df):
    df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], inplace=True)
    return df

def ordinal_encoding(df):
    df = df.replace({'BusinessTravel': {'Non-Travel':1,
                                    'Travel_Rarely':2,
                                    'Travel_Frequently':3}
                 })
    return df
    
