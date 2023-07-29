import numpy as np
from scipy import stats
from evidently.metrics import DataDriftTable
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnCorrelationsMetric
from evidently.report import Report
# from auto_statstest import StatsTest
# import win32com.client

class ModelMonitoring():
    def __init__(self, train_df):
        self.numerical_columns = self.get_numerical_columns(train_df)
        self.categorical_columns = self.get_categorical_columns(train_df)
        self.stat_test_foreach_column = {}
        self.threshold_foreach_column_stat_test = {}
    
    def get_numerical_columns(self, df):
        list_column_names = list(df.columns)
        numerical_list = []
        for col in list_column_names:
            if ((df[col].dtype in [np.int64,float])):
                numerical_list.append(col)
                continue
        return numerical_list

    def get_categorical_columns(self, df):
        list_column_names = list(df.columns)
        categorical_list = []
        for col in list_column_names:
            if ((df[col].dtype in [object, str, 'category'])):
                categorical_list.append(col)
        return categorical_list
    
    def categorical_stat_test_algo(self, df, categorical_columns, num_of_rows, column_dictionary):
        if (num_of_rows <= 1000):
            for col in categorical_columns:
                if ((df[col].nunique()) > 2):
                    column_dictionary.update({col:'chisquare'})
                else:
                    column_dictionary.update({col:'z'})
        else:
            for col in categorical_columns:
                column_dictionary.update({col:'jensenshannon'})
        return column_dictionary
    
    def numerical_stat_test_algo(self, df, numerical_columns, num_of_rows, column_dictionary):
        if (num_of_rows <= 1000):
            for col in numerical_columns:
                column_dictionary.update({col:'ks'})
        else:
            for col in numerical_columns:
                res = 0
                if (num_of_rows < 5000):
                    res = stats.shapiro(df[col])
                else:
                    res = stats.anderson(df[col])
                if (res.statistic > 0.7):
                    column_dictionary.update({col:'t_test'})
                else:
                    column_dictionary.update({col:'wasserstein'})
        return column_dictionary
    
    def set_stat_test_foreach_column(self, test_df, customise = False):
        customise = customise
        print(f"customise in set_stat_test_foreach_column : {customise}")
        if (customise == 'False'):
            numerical_column_dictionary = {}
            categorical_column_dictionary = {}
            numerical_column_dictionary = self.numerical_stat_test_algo(test_df, self.numerical_columns, len(test_df), numerical_column_dictionary)
            categorical_column_dictionary = self.categorical_stat_test_algo(test_df, self.categorical_columns, len(test_df), categorical_column_dictionary)
            self.stat_test_foreach_column = {**categorical_column_dictionary, **numerical_column_dictionary}
            # print(self.stat_test_foreach_column)
        if (customise == 'True'):
            columns_list = test_df.columns.tolist()
            stat_test_list = ['anderson', 'chisquare', 'cramer_von_mises', 'ed', 'es', 'fisher_exact', 'g_test',
                              'hellinger', 'jensenshannon', 'kl_div', 'ks', 'mannw', 'emperical_mmd', 'psi', 't_test', 
                              'TVD', 'wasserstein', 'z']
            print("The available stats tests are: \n")
            print("For categorical: ")
            print("chisquare, z, fisher_exact, g_test, TVD")
            print("For numerical: ")
            print("ks, wasserstein, anderson, cramer_von_mises, mannw, ed, es, t_test, emperical_mmd")
            print("For both categorical and numerical: ")
            print("kl_div, psi, jensenshannon, hellinger")
            print("for more information on the stats test, please refer to: \n")
            print("https://docs.evidentlyai.com/user-guide/customization/options-for-statistical-tests")
            for col in columns_list:
                stat_test_choice = input(f"for column {col}, input your stat test")
                if stat_test_choice not in stat_test_list:
                    print("stat test currently not available in this version of evidentlyAI")
                    break
                else:
                    self.stat_test_foreach_column.update({col : stat_test_choice})
                stat_test_threshold = input(f"for column {col}, input your stat test threshold")
                self.threshold_foreach_column_stat_test.update({col : stat_test_threshold})
    
    def feature_drift_report(self, train_df, test_df, format):
        print("do you want to customise the stat test for each column?")
        print("if yes, input True, else input False")
        customise = input("Choice:")
        print(f"customise in feature_drift_report : {customise}")
        self.set_stat_test_foreach_column(test_df, customise)
        feature_drift_report = Report(metrics = [
            DataDriftTable(per_column_stattest=self.stat_test_foreach_column,
                           per_column_stattest_threshold=self.threshold_foreach_column_stat_test),
        ])
        feature_drift_report.run(reference_data=train_df, current_data=test_df)
        if format == 'html':
            feature_drift_report.save_html('../html_reports/feature_drift_report.html')
        else:
            feature_drift_report.save_json('../json_reports/feature_drift_report.json')
        return feature_drift_report.as_dict()

    def prediction_drift_report(self, train_df, test_df, format):
        print("The available stats tests are: \n")
        print("For categorical: ")
        print("chisquare, z, fisher_exact, g_test, TVD")
        print("For numerical: ")
        print("ks, wasserstein, anderson, cramer_von_mises, mannw, ed, es, t_test, emperical_mmd")
        print("For both categorical and numerical: ")
        print("kl_div, psi, jensenshannon, hellinger")
        stat_test = input(f"What is your stat test of choice?")
        stat_test_threshold = input(f"What is the threshold for your stat test?")
        prediction_drift_report = Report(metrics=[
            ColumnDriftMetric(column_name='prediction', stattest=stat_test, 
                              stattest_threshold=stat_test_threshold),
        ])
        prediction_drift_report.run(reference_data=train_df, current_data=test_df)
        if format == 'html':
            prediction_drift_report.save_html('../html_reports/prediction_drift_report.html')
        else:
            prediction_drift_report.save_json('../json_reports/prediction_drift_report.json')
        return prediction_drift_report.as_dict()
    




    def check_schema(self, train_df, test_df):
        train_column_list = train_df.columns.tolist()
        test_column_list = test_df.columns.tolist()
        train_set = set(train_column_list)
        test_set = set(test_column_list)
        if (train_set == test_set):
            print(f"train and test dataset have the same features")
            return 1
        else:
            print(f"train and test dataset do not have the same features")
            return 0

    def check_schema_postprocessing(self, processed_train_df, processed_test_df):
        processed_train_column_list = processed_train_df.columns.tolist()
        processed_test_column_list = processed_test_df.columns.tolist()
        processed_train_set = set(processed_train_column_list)
        processed_test_set = set(processed_test_column_list)
        if (processed_train_set == processed_test_set):
            print(f"train and test dataset have the same processed features")
            return 1
        else:
            print(f"train and test dataset do not have the same processed features")
            raise 0    

    def check_data_types(self, train_df, test_df):
        columns_list = train_df.columns.tolist()
        for col in columns_list:
            if (train_df[col].dtype == test_df[col].dtype):
                # print(f"Data Type of {col} from both datasets are the same")
                continue
            else:
                print(f"Column name is : {col}")
                raise TypeError(f"Data Type of {col} in production does not match with training")
        print(f"The data types for all columns from both datasets are the same")

    def replace_column_names(self, df):   
        df.rename(columns=lambda s: s.replace(" ", "_" ), inplace=True)

    # def notify_schema_change(self, train_df, test_df):
    #     check_schema_flag = self.check_schema(train_df=train_df, test_df=test_df)
    #     if check_schema_flag == 1:
    #         return
    #     else:
    #         # toast = ToastNotifier()
    #         # toast.show_toast(
    #         #     "Schema Change",
    #         #     "There is a change in the schema, do consider retraining for bettter model",
    #         #     duration = 20,
    #         #     icon_path=None,
    #         #     threaded=True,
    #         # )
    #         ol = win32com.client.Dispatch('Outlook.Application')
    #         olmailitem = 0x0
    #         newmail=ol.CreateItem(olmailitem)
    #         newmail.Subject= 'Change in schema'
    #         newmail.To='hanhuateo@gmail.com'
    #         # newmail.CC='xyz@gmail.com'
    #         newmail.Body= 'Hello, there is a change in the incoming of features for production data, retraining will commence.'
    #         newmail.Send()
    #         # insert retrain model here
    #     return 

    def handle_bad_data(self, df):
        df = df.replace(['?', '-'], np.nan)
        return df

    def data_check(self, train_df, test_df, processed_train_df, processed_test_df):
        self.replace_column_names(test_df)
        test_df = self.handle_bad_data(test_df)
        check_schema_flag = self.check_schema(train_df, test_df)
        check_data_type_flag = self.check_data_types(train_df, test_df)
        check_processed_schema_flag = self.check_schema_postprocessing(processed_train_df, processed_test_df)
        if check_data_type_flag == 0:
            print(f"there is a problem with the data types of the incoming data")
        if check_processed_schema_flag == 0:
            print(f"there is a problem with the preprocessing of the incoming data")
        if check_schema_flag == 0:
            print(f"there is a change in the schema of the incoming data")