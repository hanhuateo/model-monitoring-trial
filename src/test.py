from monitoring.evidently import *
import pandas as pd
from algo.algo import data_type_algo, categorical_stat_test_algo, numerical_stat_test_algo
from monitoring.evidently import DataDriftTable, Report

def merge_dictionaries(dic1, dic2):
    dic3 = dic1.update(dic2)
    return dic3

X_combined = pd.read_csv('../data/split_data/X_combined.csv')
X_current = pd.read_csv('../data/split_data/X_current.csv')
numerical_list, nominal_list, ordinal_list = data_type_algo(X_combined)
print("numerical list : {}".format(numerical_list))
print("nominal list : {}".format(nominal_list))
print("ordinal list : {}".format(ordinal_list))

categorical_column_dictionary = {}
numerical_column_dictionary = {}

categorical_column_dictionary = categorical_stat_test_algo(X_combined, nominal_list, len(X_combined), categorical_column_dictionary)
# numerical_column_dictionary = numerical_stat_test_algo(X_combined, numerical_list, len(X_combined), numerical_column_dictionary)
numerical_column_dictionary = numerical_stat_test_algo(X_combined, ordinal_list, len(X_combined), numerical_column_dictionary)

# print(categorical_column_dictionary)
# print(numerical_column_dictionary)
column_dictionary = {**numerical_column_dictionary, **categorical_column_dictionary}
# column_dictionary = merge_dictionaries(categorical_column_dictionary, numerical_column_dictionary)
# print(column_dictionary)
data_drift_report = Report(metrics=[
    DataDriftTable(per_column_stattest=column_dictionary)
])
data_drift_report.run(reference_data=X_combined, current_data=X_current)
data_drift_report.save_html('./monitoring/reports/data_drift_report.html')