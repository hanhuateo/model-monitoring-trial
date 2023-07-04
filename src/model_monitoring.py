import numpy as np
from scipy import stats
from evidently.metrics import DataDriftTable
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, ColumnCorrelationsMetric, TargetByFeaturesTable

class ModelMonitoring():
    def __init__(self):
        self.numerical_columns = []
        self.categorical_columns = []
        # self.ground_truth_column = [] 
        # self.train_prediction_column = [] 
        self.stat_test_foreach_column = {}
        self.column_mapping = ColumnMapping()
        
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
        
    def feature_drift_report(self, train_df, test_df):
        self.numerical_columns = self.get_numerical_columns(test_df)
        self.categorical_columns = self.get_categorical_columns(test_df)
        self.column_mapping.numerical_features = self.numerical_columns
        self.column_mapping.categorical_features = self.categorical_columns
        self.column_mapping.task = 'classification'
        numerical_column_dictionary = {}
        categorical_column_dictionary = {}
        numerical_column_dictionary = self.numerical_stat_test_algo(test_df, self.numerical_columns, len(test_df), numerical_column_dictionary)
        categorical_column_dictionary = self.categorical_stat_test_algo(test_df, self.categorical_columns, len(test_df), categorical_column_dictionary)
        self.stat_test_foreach_column = {**categorical_column_dictionary, **numerical_column_dictionary}
        feature_drift_report = Report(metrics = [
            DataDriftTable(per_column_stattest=self.stat_test_foreach_column),
            # TargetDriftPreset(),
            ColumnDriftMetric(column_name='prediction'),
            ColumnDriftMetric(column_name='target'),
            ColumnCorrelationsMetric(column_name='prediction'),
            ColumnCorrelationsMetric(column_name='target'),
            
        ])
        feature_drift_report.run(reference_data=train_df, current_data=test_df)
        feature_drift_report.save_html('../reports/feature_drift_report.html')

    def target_drift_report(self, train_df, test_df):
        target_drift_report = Report(metrics=[
            TargetDriftPreset(),
        ])
        target_drift_report.run(reference_data=train_df, current_data=test_df)
        target_drift_report.save_html('../reports/feature_drift_report.html')

    def set_ground_truth(self, ground_truth_column):
        self.ground_truth_column = ground_truth_column