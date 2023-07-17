import numpy as np
from scipy import stats
from evidently.metrics import DataDriftTable
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnCorrelationsMetric
from evidently.metric_preset import DataQualityPreset
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.test_preset import DataQualityTestPreset, DataStabilityTestPreset

class ModelMonitoring():
    def __init__(self, train_df):
        self.numerical_columns = self.get_numerical_columns(train_df)
        self.categorical_columns = self.get_categorical_columns(train_df)
        self.stat_test_foreach_column = {}
        self.columns_list = self.get_columns_list(train_df)
        
    def get_columns_list(self, df):
        columns_list = df.columns.tolist()
        return columns_list
    
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
            feature_drift_report.save_html('../reports/feature_drift_report.html')
        else:
            feature_drift_report.save_json('../reports/feature_drift_report.json')
        return feature_drift_report.as_dict()

    def prediction_drift_report(self, train_df, test_df, format):
        prediction_drift_report = Report(metrics=[
            ColumnDriftMetric(column_name='prediction'),
            ColumnCorrelationsMetric(column_name='prediction'),
        ])
        prediction_drift_report.run(reference_data=train_df, current_data=test_df)
        if format == 'html':
            prediction_drift_report.save_html('../reports/prediction_drift_report.html')
        else:
            prediction_drift_report.save_json('../reports/prediction_drift_report.json')
        return prediction_drift_report.as_dict()
    
    def data_quality_report(self, train_df, test_df, format):
        data_quality_report = Report(metrics=[
            DataQualityPreset()
        ])
        data_quality_report.run(reference_data=train_df, current_data=test_df)
        if format == 'html':
            data_quality_report.save_html('../reports/data_quality_report.html')
        else:
            data_quality_report.save_json('../reports/data_quality_report.json')
        return data_quality_report.as_dict()

    def data_quality_test_suite(self, train_df, test_df, format):
        data_quality_test_suite = TestSuite(tests=[
            DataQualityTestPreset(),
        ])
        data_quality_test_suite.run(reference_data=train_df, current_data=test_df)
        if format == 'html':
            data_quality_test_suite.save_html('../reports/data_quality_test_suite.html')
        else:
            data_quality_test_suite.save_json('../reports/data_quality_test_suite.json')
        return data_quality_test_suite.as_dict()
    
    def check_schema(self, train_df, test_df):
        train_column_list = train_df.columns.tolist()
        test_column_list = test_df.columns.tolist()
        train_set = set(train_column_list)
        test_set = set(test_column_list)
        if (train_set == test_set):
            return 1
        else:
            raise Exception("Data Schema has changed, do consider retraining model on new schema")
        
    # def check_one_hot_encoding(self, train_df, test_df, processed_test_df):
    #     same_columns_flag = self.check_schema(train_df, test_df)
    #     if (same_columns_flag == 0):
    #         raise Exception("Train and Test dataframes do not have the same columns")
    #     else:
    #         for col in self.categorical_columns:
    