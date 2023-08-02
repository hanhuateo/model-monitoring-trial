import numpy as np
from scipy import stats
from evidently.metrics import DataDriftTable
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
import json
# import win32com.client
from joblib import load
from evidently import ColumnMapping

class ModelMonitoring():
    def __init__(self):
        self.columns_list = load('./preprocessor/training_column_names_list.joblib')
        self.numerical_columns = load('./preprocessor/training_numerical_columns.joblib')
        self.categorical_columns = load('./preprocessor/training_categorical_columns.joblib')
        self.stat_test_foreach_column = {}
        self.stat_test_threshold_foreach_column = {}
        self.column_mapping = ColumnMapping()
        self.column_mapping.numerical_features = self.numerical_columns
        self.column_mapping.categorical_features = self.categorical_columns
    
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
    
    def set_stat_test_foreach_column(self, test_df):
        numerical_column_dictionary = {}
        categorical_column_dictionary = {}
        numerical_column_dictionary = self.numerical_stat_test_algo(test_df, self.numerical_columns, len(test_df), numerical_column_dictionary)
        categorical_column_dictionary = self.categorical_stat_test_algo(test_df, self.categorical_columns, len(test_df), categorical_column_dictionary)
        self.stat_test_foreach_column = {**categorical_column_dictionary, **numerical_column_dictionary}

    def set_stat_test_threshold_foreach_column(self):
        with open('config.json', 'r') as f:
            data = json.load(f)
        self.set_stat_test_threshold_foreach_column = data

    def save_data_to_json(self, data, file_name):
        with open(file_name, 'w') as f:
            json.dump(data, f)

    def write_to_config(self):
        data = {}
        for col in self.columns_list:
            data.update({col: 0.05})
        self.save_data_to_json(data, 'config.json')

    def feature_drift_report(self, train_df, incoming_df, format):
        self.set_stat_test_foreach_column(incoming_df)
        self.set_stat_test_threshold_foreach_column()
        feature_drift_report = Report(metrics = [
            DataDriftTable(per_column_stattest=self.stat_test_foreach_column,
                        per_column_stattest_threshold=self.stat_test_threshold_foreach_column,
                        ),
        ])
        feature_drift_report.run(reference_data=train_df, current_data=incoming_df, column_mapping=self.column_mapping)
        if format == 'html':
            feature_drift_report.save_html('../html_reports/feature_drift_report.html')
        else:
            feature_drift_report.save_json('../json_reports/feature_drift_report.json')
            

    def prediction_drift_report(self, train_df, incoming_df, stat_test, stat_test_threshold, format):
        prediction_drift_report = Report(metrics=[
            ColumnDriftMetric(column_name='prediction',
                              stattest=stat_test,
                              stattest_threshold=stat_test_threshold),
        ])
        prediction_drift_report.run(reference_data=train_df, current_data=incoming_df)
        if format == 'html':
            prediction_drift_report.save_html('../html_reports/prediction_drift_report.html')
        else:
            prediction_drift_report.save_json('../json_reports/prediction_drift_report.json')
    
    # def check_for_drift(self, option):
    #     """
    #     The function `check_for_drift` checks for feature drift or prediction drift and sends an email
    #     notification if drift is detected.
        
    #     :param option: The `option` parameter is used to specify whether to check for feature drift or
    #     prediction drift. It can have two possible values: 'feature' or 'prediction'
    #     """
    #     columns_list = self.columns_list
    #     feature_drift_list = []
    #     if option == 'feature':
    #         f = open('../json_reports/feature_drift_report.json')
    #         json_report = json.load(f)
    #         for col in columns_list:
    #             if (json_report['metrics'][0]['result']['drift_by_columns'][col]['drift_score'] < json_report['metrics'][0]['result']['drift_by_columns'][col]['stattest_threshold']):
    #                 feature_drift_list.append(col)

    #         if bool(feature_drift_list):
    #             print(f"the list of features that has drifted : {feature_drift_list}")
    #             ol=win32com.client.Dispatch("outlook.application")
    #             olmailitem=0x0 #size of the new email
    #             newmail=ol.CreateItem(olmailitem)
    #             newmail.Subject= 'Feature Drift'
    #             newmail.To='hanhuateo@gmail.com'
    #             # newmail.CC='xyz@gmail.com'
    #             newmail.Body=f"Hello, these are the columns that have drifted : {feature_drift_list}"
    #             newmail.Send()
    #         f.close()

    #     if option == 'prediction':
    #         f = open('../json_reports/prediction_drift_report.json')
    #         json_report = json.load(f)
    #         if (json_report['metrics'][0]['result']['drift_score'] < json_report['metrics'][0]['result']['stattest_threshold']):
    #             ol=win32com.client.Dispatch("outlook.application")
    #             olmailitem=0x0
    #             newmail = ol.CreateItem(olmailitem)
    #             newmail.Subject = 'Prediction Drift'
    #             newmail.To='hanhuateo@gmail.com'
    #             # newmail.CC='xyz@gmail.com'
    #             newmail.Body = "Hello, prediction column has drifted"
    #             newmail.Send()

    def check_schema(self, train_df, incoming_df):
        train_column_list = train_df.columns.tolist()
        test_column_list = incoming_df.columns.tolist()
        train_set = set(train_column_list)
        test_set = set(test_column_list)
        if (train_set != test_set):
            raise Exception(f"the two datasets do not have the same features")

    def check_data_types(self, train_df, incoming_df):
        columns_list = train_df.columns.tolist()
        for col in columns_list:
            if (train_df[col].dtype != incoming_df[col].dtype):
                raise TypeError(f"Data Type of {col} in production does not match with training")

    def data_check(self, train_df, incoming_df, processed_train_df, processed_incoming_df):
        self.check_data_types(train_df, incoming_df)
        self.check_schema(train_df, incoming_df)
        self.check_schema(processed_train_df, processed_incoming_df)

    def get_feature_importance_mapping(self):
        RF_clf = load('./model/RF_clf.joblib')
        feature_importance = RF_clf.feature_importances_
        print(f"feature importances : {feature_importance}, type : {type(feature_importance)}")
        column_transformer = load('./preprocessor/column_transformer.pkl')
        feature_names = column_transformer.get_feature_names_out()
        print(f"feature names : {feature_names}, type : {type(feature_names)}")
        feature_importance_mapping = {}
        for name, importance in zip(feature_names, feature_importance):
            feature_importance_mapping.update({name : importance})

        print(f"feature importance mapping : {feature_importance_mapping}")
        return feature_importance_mapping