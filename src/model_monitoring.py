import numpy as np
from scipy import stats
from evidently.metrics import DataDriftTable
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnCorrelationsMetric
from evidently.metric_preset import DataQualityPreset
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.test_preset import DataQualityTestPreset, DataStabilityTestPreset
from win10toast import ToastNotifier
import time

class ModelMonitoring():
    def __init__(self, train_df):
        self.numerical_columns = self.get_numerical_columns(train_df)
        self.categorical_columns = self.get_categorical_columns(train_df)
        self.stat_test_foreach_column = {}
    
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
        
    def feature_drift_report(self, train_df, test_df, format):
        numerical_column_dictionary = {}
        categorical_column_dictionary = {}
        numerical_column_dictionary = self.numerical_stat_test_algo(test_df, self.numerical_columns, len(test_df), numerical_column_dictionary)
        categorical_column_dictionary = self.categorical_stat_test_algo(test_df, self.categorical_columns, len(test_df), categorical_column_dictionary)
        self.stat_test_foreach_column = {**categorical_column_dictionary, **numerical_column_dictionary}
        feature_drift_report = Report(metrics = [
            # DataDriftTable(per_column_stattest=self.stat_test_foreach_column),
            DataDriftTable()
        ])
        feature_drift_report.run(reference_data=train_df, current_data=test_df)
        if format == 'html':
            feature_drift_report.save_html('../html_reports/feature_drift_report.html')
        else:
            feature_drift_report.save_json('../json_reports/feature_drift_report.json')
        return feature_drift_report.as_dict()

    def prediction_drift_report(self, train_df, test_df, format):
        prediction_drift_report = Report(metrics=[
            ColumnDriftMetric(column_name='prediction'),
            ColumnCorrelationsMetric(column_name='prediction'),
        ])
        prediction_drift_report.run(reference_data=train_df, current_data=test_df)
        if format == 'html':
            prediction_drift_report.save_html('../html_reports/prediction_drift_report.html')
        else:
            prediction_drift_report.save_json('../json_reports/prediction_drift_report.json')
        return prediction_drift_report.as_dict()
    
    def data_quality_report(self, train_df, test_df, format):
        data_quality_report = Report(metrics=[
            DataQualityPreset()
        ])
        data_quality_report.run(reference_data=train_df, current_data=test_df)
        if format == 'html':
            data_quality_report.save_html('../html_reports/data_quality_report.html')
            return data_quality_report
        else:
            data_quality_report.save_json('../json_reports/data_quality_report.json')
            return data_quality_report.as_dict()

    def data_quality_test_suite(self, train_df, test_df, format):
        data_quality_test_suite = TestSuite(tests=[
            DataQualityTestPreset(),
        ])
        data_quality_test_suite.run(reference_data=train_df, current_data=test_df)
        if format == 'html':
            data_quality_test_suite.save_html('../html_reports/data_quality_test_suite.html')
        else:
            data_quality_test_suite.save_json('../json_reports/data_quality_test_suite.json')
        return data_quality_test_suite.as_dict()
    
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
    
    def check_processing_validity(self, train_df, test_df, processed_train_df, processed_test_df):
        check_schema_flag = self.check_schema(train_df, test_df)
        if (check_schema_flag == 1):
            pass
        else:
            print("training and production data schema does not match")
            return
        categorical_columns = self.categorical_columns
        total_num_of_columns = len(train_df.columns)
        print(f"total number of columns : {total_num_of_columns}")
        total_num_of_processed_columns = len(processed_train_df.columns)
        print(f"total number of processed columns : {total_num_of_processed_columns}")
        count_of_columns = total_num_of_columns
        for col in categorical_columns:
            number_of_unique_values = train_df[col].nunique()
            count_of_columns = count_of_columns + number_of_unique_values - 1
            print(f"count of columns : {count_of_columns}")
        
        if (count_of_columns == total_num_of_processed_columns):
            return 1
        else:
            return 0

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

    def handle_missing_values(self, data_quality_report_json, method, test_df):
        nans_by_columns_dict = data_quality_report_json['metrics'][0]['result']['current']['nans_by_columns']
        columns_with_missing_values = []
        # get columns with missing values
        for key in nans_by_columns_dict.keys():
            if nans_by_columns_dict[key] > 0:
                columns_with_missing_values.append(key)
            if nans_by_columns_dict[key] == 0:
                continue
        
        # drop records with missing values
        if method == 'delete':
            test_df = test_df.dropna(subset=[columns_with_missing_values])
        
        # impute the records with missing values
        if method == 'impute':
            for col in columns_with_missing_values:
                if test_df[col].dtype in [np.int64, float]:
                    test_df[col].fillna(test_df[col].mean(), inplace=True)
                if test_df[col].dtype in [object, 'category']:
                    test_df[col].fillna(test_df[col].mode()[0], inplace=True)
        
        return test_df

    def notify_schema_change(self, train_df, test_df):
        check_schema_flag = self.check_schema(train_df=train_df, test_df=test_df)
        if check_schema_flag == 1:
            return
        else:
            toast = ToastNotifier()
            toast.show_toast(
                "Schema Change",
                "There is a change in the schema, do consider retraining for bettter model",
                duration = 20,
                icon_path=None,
                threaded=True,
            )
            while toast.notification_active(): time.sleep(0.1)

    def check_bad_data(self, df):
        df = df.replace(['?', '-'], np.nan)
        return df
    
    def check(self):
        pass