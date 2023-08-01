import numpy as np
from scipy import stats
from evidently.metrics import DataDriftTable
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
import json
# import win32com.client
from joblib import load

class ModelMonitoring():
    def __init__(self):
        """
        The function initializes the object with the numerical and categorical columns of a given
        dataframe, as well as empty dictionaries for statistical tests and their thresholds.
        
        :param train_df: The train_df parameter is a pandas DataFrame that contains the training data
        for your machine learning model. It is used to initialize the object and extract the numerical
        and categorical columns from the DataFrame
        """
        self.columns_list = load('./preprocessor/training_column_names_list.joblib')
        self.numerical_columns = load('./preprocessor/training_numerical_columns.joblib')
        self.categorical_columns = load('./preprocessor/training_categorical_columns.joblib')
        self.stat_test_foreach_column = {}
        self.stat_test_threshold_foreach_column = {}
    
    
    def categorical_stat_test_algo(self, df, categorical_columns, num_of_rows, column_dictionary):
        """
        The function `categorical_stat_test_algo` determines the appropriate statistical test to use for
        each categorical column in a dataframe based on the number of rows in the dataframe and the 
        number of unique values in the column.
        
        :param df: The dataframe containing the data
        :param categorical_columns: A list of column names in the dataframe that contain categorical
        variables
        :param num_of_rows: The parameter "num_of_rows" represents the number of rows in the dataframe
        "df"
        :param column_dictionary: The `column_dictionary` parameter is an empty dictionary that maps each
        categorical column in the dataframe to a statistical test algorithm. The keys of the dictionary
        are the column names, and the values are the names of the statistical test algorithms
        :return: the updated column_dictionary.
        """
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
        """
        The function `numerical_stat_test_algo` determines the appropriate statistical test to use for
        each numerical column in a dataframe based on the number of rows and the results of Shapiro-Wilk
        or Anderson-Darling tests.
        
        :param df: The parameter "df" is a pandas DataFrame that contains the data for which you want to
        perform the statistical tests
        :param numerical_columns: The numerical_columns parameter is a list of column names in the
        dataframe that contain numerical data
        :param num_of_rows: The parameter "num_of_rows" represents the number of rows in the dataframe
        "df"
        :param column_dictionary: The `column_dictionary` parameter is an empty dictionary that maps column
        names to statistical test names. The function updates this dictionary based on the number of
        rows in the dataframe and the type of statistical test to be performed on each numerical column
        :return: the updated column_dictionary.
        """
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
        """
        The `set_stat_test_foreach_column` function sets statistical tests for each column in a given
        dataframe, either using default tests or allowing customization.
        
        :param test_df: The `test_df` parameter is a DataFrame that contains the data on which the
        statistical tests will be performed
        :param customise: The `customise` parameter is a boolean flag that determines whether to use the
        default statistical tests or allow customization of the tests for each column. If `customise` is set
        to `False`, the function will use the default statistical tests. If `customise` is set to `True`,,
        defaults to False (optional)
        """
        numerical_column_dictionary = {}
        categorical_column_dictionary = {}
        numerical_column_dictionary = self.numerical_stat_test_algo(test_df, self.numerical_columns, len(test_df), numerical_column_dictionary)
        categorical_column_dictionary = self.categorical_stat_test_algo(test_df, self.categorical_columns, len(test_df), categorical_column_dictionary)
        self.stat_test_foreach_column = {**categorical_column_dictionary, **numerical_column_dictionary}

    def set_stat_test_threshold_foreach_column(self, incoming_df):
        with open("config.json", "r") as jsonfile:
            data = json.load(jsonfile)
        print(f"categorical p-value threshold is : {data['categorical']['p-value']}")
        print(f"categorical divergence-distance threshold is : {data['categorical']['divergence-distance']}")
        print(f"numerical p-value threshold is : {data['numerical']['p-value']}")
        print(f"numerical divergence-distance threshold is : {data['numerical']['divergence-distance']}")
    
    def feature_drift_report(self, train_df, incoming_df, format):
        """
        The function `feature_drift_report` generates a report on feature drift between a training
        dataset and a test dataset, allowing for customization of statistical tests for each column.
        
        :param train_df: The `train_df` parameter is the training dataset that is used as the reference
        data for the feature drift analysis. It should be a pandas DataFrame object
        :param test_df: The `test_df` parameter is a DataFrame that contains the data for which you want
        to generate the feature drift report. This DataFrame should have the same columns as the
        `train_df` DataFrame, which is used as the reference data for the report
        :param format: The "format" parameter is used to specify the format in which the feature drift
        report should be saved. It can take two values: 'html' or 'json'. If 'html' is chosen, the
        report will be saved as an HTML file. If 'json' is chosen, the report will be saved as a JSON file.
        :return: the feature drift report as a dictionary.
        """
        self.set_stat_test_foreach_column(incoming_df)
        self.set_stat_test_threshold_foreach_column(incoming_df)
        feature_drift_report = Report(metrics = [
            DataDriftTable(per_column_stattest=self.stat_test_foreach_column,
                           per_column_stattest_threshold=self.stat_test_threshold_foreach_column),
        ])
        feature_drift_report.run(reference_data=train_df, current_data=incoming_df)
        if format == 'html':
            feature_drift_report.save_html('../html_reports/feature_drift_report.html')
        else:
            feature_drift_report.save_json('../json_reports/feature_drift_report.json')

    def prediction_drift_report(self, train_df, test_df, format):
        """
        The function `prediction_drift_report` generates a prediction drift report using statistical
        tests and saves it in either HTML or JSON format.
        
        :param train_df: The `train_df` parameter is the training dataset that is used as the reference
        data for the prediction drift report. It should be a pandas DataFrame containing the training
        data
        :param test_df: The `test_df` parameter is the DataFrame containing the test data that you want
        to evaluate for prediction drift
        :param format: The "format" parameter is used to specify the format in which the prediction
        drift report should be saved. It can take two values: 'html' or 'json'. If 'html' is chosen, the
        report will be saved as an HTML file. If 'json' is chosen, the report
        :return: the prediction drift report as a dictionary.
        """
        prediction_drift_report = Report(metrics=[
            ColumnDriftMetric(column_name='prediction'),
        ])
        prediction_drift_report.run(reference_data=train_df, current_data=test_df)
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
        """
        The function checks if two dataframes have the same set of columns.
        
        :param train_df: The train_df parameter is a pandas DataFrame that represents the training
        dataset. It contains the features (columns) that will be used to train a machine learning model
        :param test_df: The `test_df` parameter is a DataFrame that represents the test dataset. It
        contains the data that will be used to evaluate the performance of a machine learning model
        :return: either 1 or 0, depending on whether the train and test datasets have the same features
        or not.
        """
        train_column_list = train_df.columns.tolist()
        test_column_list = incoming_df.columns.tolist()
        train_set = set(train_column_list)
        test_set = set(test_column_list)
        if (train_set != test_set):
            raise Exception(f"the two datasets do not have the same features")

    def check_data_types(self, train_df, incoming_df):
        """
        The function checks if the data types of columns in the training and test datasets are the same,
        and raises an error if they are not.
        
        :param train_df: The training dataset, which is a pandas DataFrame containing the data used to
        train a machine learning model
        :param test_df: The `test_df` parameter is a DataFrame that contains the test data. It is used
        to compare the data types of columns with the corresponding columns in the `train_df` DataFrame
        """
        columns_list = train_df.columns.tolist()
        for col in columns_list:
            if (train_df[col].dtype != incoming_df[col].dtype):
                raise TypeError(f"Data Type of {col} in production does not match with training")

    def data_check(self, train_df, incoming_df, processed_train_df, processed_incoming_df):
        """
        The `data_check` function checks the schema and data types of the incoming data and prints error
        messages if there are any issues.
        
        :param train_df: The `train_df` parameter is a DataFrame that represents the training data. It
        contains the features (input variables) and the target variable (output variable) for training a
        machine learning model
        :param test_df: test_df is the test dataset that needs to be checked for data quality and
        consistency
        :param processed_train_df: The parameter `processed_train_df` is a DataFrame that represents the
        processed training data. It is the result of applying some preprocessing steps to the original
        training data
        :param processed_test_df: The parameter `processed_test_df` is a DataFrame that represents the
        processed or transformed version of the test data. It is the output of some preprocessing steps
        applied to the test data before using it for further analysis or modeling
        """
        self.check_data_types(train_df, incoming_df)
        self.check_schema(train_df, incoming_df)
        self.check_schema(processed_train_df, processed_incoming_df)
        