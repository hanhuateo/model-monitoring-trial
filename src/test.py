from monitoring.evidently import *
import pandas as pd
from algo.algo import data_type_algo, categorical_stat_test_algo, numerical_stat_test_algo

X_combined = pd.read_csv('../data/split_data/X_combined.csv')
X_current = pd.read_csv('../data/split_data/X_current.csv')
numerical_list, nominal_list, ordinal_list = data_type_algo(X_combined)
print("numerical list : {}".format(numerical_list))
print("nominal list : {}".format(nominal_list))
print("ordinal list : {}".format(ordinal_list))

column_dictionary = {}

# column_dictionary = categorical_stat_test_algo(X_combined, nominal_list, len(X_combined), column_dictionary)

column_dictionary = numerical_stat_test_algo(X_combined, numerical_list, len(X_combined), column_dictionary)
# column_dictionary = numerical_stat_test_algo(X_combined, ordinal_list, len(X_combined), column_dictionary)

print(column_dictionary)