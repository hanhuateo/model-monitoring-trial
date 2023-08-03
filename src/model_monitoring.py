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
        """
        The function initializes several variables and loads preprocessed data for further use.
        """
        self.columns_list = load('./preprocessor/training_column_names_list.joblib')
        self.numerical_columns = load('./preprocessor/training_numerical_columns.joblib')
        self.categorical_columns = load('./preprocessor/training_categorical_columns.joblib')
        self.stat_test_foreach_column = {}
        self.stat_test_threshold_foreach_column = {}
        self.column_mapping = ColumnMapping()
        self.column_mapping.numerical_features = self.numerical_columns
        self.column_mapping.categorical_features = self.categorical_columns
    
    def categorical_stat_test_algo(self, df, categorical_columns, num_of_rows, column_dictionary):
        """
        The function `categorical_stat_test_algo` determines the appropriate statistical test to use for
        each categorical column in a dataframe based on the number of unique values and the number of rows
        in the dataframe.
        
        :param df: The df parameter is a pandas DataFrame that contains the data you want to perform the
        statistical test on
        :param categorical_columns: A list of column names in the dataframe that contain categorical
        variables
        :param num_of_rows: The parameter "num_of_rows" represents the number of rows in the dataframe "df"
        :param column_dictionary: The `column_dictionary` is a dictionary that maps each categorical column
        in the dataframe to a statistical test algorithm. The keys of the dictionary are the column names,
        and the values are the corresponding statistical test algorithms
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
        The function `numerical_stat_test_algo` determines the appropriate statistical test to use for each
        numerical column in a dataframe based on the number of rows and the results of the Shapiro-Wilk or
        Anderson-Darling tests.
        
        :param df: The parameter "df" is a pandas DataFrame that contains the data you want to perform
        statistical tests on
        :param numerical_columns: A list of column names in the dataframe that contain numerical data
        :param num_of_rows: The parameter "num_of_rows" represents the number of rows in the dataframe "df"
        :param column_dictionary: The `column_dictionary` is a dictionary that maps each numerical column in
        the dataframe to a statistical test. The keys of the dictionary are the column names, and the values
        are the corresponding statistical test to be performed on that column
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
        The function `set_stat_test_foreach_column` calculates statistical tests for each column in a given
        dataframe and stores the results in a dictionary.
        
        :param test_df: The `test_df` parameter is a pandas DataFrame that contains the data on which you
        want to perform statistical tests
        """
        numerical_column_dictionary = {}
        categorical_column_dictionary = {}
        numerical_column_dictionary = self.numerical_stat_test_algo(test_df, self.numerical_columns, len(test_df), numerical_column_dictionary)
        categorical_column_dictionary = self.categorical_stat_test_algo(test_df, self.categorical_columns, len(test_df), categorical_column_dictionary)
        self.stat_test_foreach_column = {**categorical_column_dictionary, **numerical_column_dictionary}

    def set_stat_test_threshold_foreach_column(self):
        """
        The function reads a JSON file called 'config.json' and sets the value of
        'set_stat_test_threshold_foreach_column' to the data in the file.
        """
        with open('config.json', 'r') as f:
            data = json.load(f)
        self.set_stat_test_threshold_foreach_column = data

    def save_data_to_json(self, data, file_name):
        """
        The function saves data to a JSON file. But in this case, this function is used to save the 
        columns of the dataframe as keys and using a standard 0.05 threshold for statistical tests as values
        which can be customised.
        
        :param data: The data parameter is the data that you want to save to a JSON file. It can be any
        valid JSON data, such as a dictionary, list, or string
        :param file_name: The file name is a string that specifies the name of the file where the data will
        be saved. It should include the file extension, such as ".json" for a JSON file
        """
        with open(file_name, 'w') as f:
            json.dump(data, f)

    def write_to_config(self):
        """
        The function `write_to_config` creates a dictionary with keys from `self.columns_list` and values of
        0.05, then saves the dictionary as JSON in a file named 'config.json'.
        """
        data = {}
        for col in self.columns_list:
            data.update({col: 0.05})
        self.save_data_to_json(data, 'config.json')

    def feature_drift_report(self, train_df, incoming_df, format):
        """
        The function `feature_drift_report` generates a report on feature drift between two datasets and
        saves it in either HTML or JSON format.
        
        :param train_df: The `train_df` parameter is the training dataset that is used as the reference data
        for the feature drift report. It contains the data that the incoming dataset will be compared
        against to detect any drift in the features
        :param incoming_df: The `incoming_df` parameter is a DataFrame that contains the incoming data for
        which you want to generate a feature drift report. This DataFrame should have the same columns as
        the `train_df` DataFrame, which is the reference data used for comparison
        :param format: The "format" parameter is a string that specifies the format in which the feature
        drift report should be saved. It can have two possible values: "html" or any other value (e.g.,
        "json")
        """
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
        """
        The `prediction_drift_report` function generates a report on the drift of predictions between a
        reference dataset and an incoming dataset, using a specified statistical test and threshold, and
        saves the report in either HTML or JSON format.
        
        :param train_df: The `train_df` parameter is the reference dataset that you use to train your model.
        It should be a pandas DataFrame containing the training data
        :param incoming_df: The `incoming_df` parameter is a DataFrame that contains the incoming data for
        which you want to generate a prediction drift report. This DataFrame should have the same structure
        as the `train_df` DataFrame, which is the reference data used to train the model
        :param stat_test: The `stat_test` parameter is the statistical test used to detect drift in the
        prediction column. It could be any statistical test such as t-test, chi-square test, or
        Kolmogorov-Smirnov test. The specific test to be used should be specified as a string
        :param stat_test_threshold: The `stat_test_threshold` parameter is a threshold value used in the
        statistical test to determine if there is a significant drift in the predictions between the
        reference data and the current data. If the p-value or test statistic exceeds this threshold, it
        indicates that there is a significant drift in the predictions
        :param format: The "format" parameter specifies the format in which the prediction drift report
        should be saved. It can have two possible values: "html" or any other value (e.g., "json"). If the
        value is "html", the report will be saved in HTML format. Otherwise, it will be saved in JSON format.
        """
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

    def check_schema(self, train_df, incoming_df):
        """
        The function `check_schema` compares the column names of two dataframes and raises an exception if
        they are not the same.
        
        :param train_df: A pandas DataFrame containing the training dataset
        :param incoming_df: The `incoming_df` parameter is a DataFrame that represents the incoming dataset
        that needs to be checked against the training dataset
        """
        train_column_list = train_df.columns.tolist()
        test_column_list = incoming_df.columns.tolist()
        train_set = set(train_column_list)
        test_set = set(test_column_list)
        if (train_set != test_set):
            raise Exception(f"the two datasets do not have the same features")

    def check_data_types(self, train_df, incoming_df):
        """
        The function `check_data_types` compares the data types of columns in two dataframes and raises a
        TypeError if any data types do not match.
        
        :param train_df: The `train_df` parameter is a DataFrame that represents the training data. It
        contains the columns and their corresponding data types that were used to train a model
        :param incoming_df: The `incoming_df` parameter is a DataFrame that contains the data that is being
        compared to the `train_df` DataFrame. It is used to check if the data types of the columns in the
        `incoming_df` match the data types of the corresponding columns in the `train_df`
        """
        columns_list = train_df.columns.tolist()
        for col in columns_list:
            if (train_df[col].dtype != incoming_df[col].dtype):
                raise TypeError(f"Data Type of {col} in production does not match with training")

    def data_check(self, train_df, incoming_df, processed_train_df, processed_incoming_df):
        """
        The function `data_check` performs various checks on the dataframes passed as arguments.
        
        :param train_df: The `train_df` parameter represents the training dataset that is used to train a
        machine learning model
        :param incoming_df: The `incoming_df` parameter is a DataFrame that contains the incoming data that
        needs to be checked against the existing data
        :param processed_train_df: The parameter "processed_train_df" is a DataFrame that represents the
        processed version of the training data. It is likely that some transformations or feature
        engineering has been applied to the original training data to create this processed version
        :param processed_incoming_df: The parameter "processed_incoming_df" is a DataFrame that represents
        the processed version of the incoming data. It is likely that this DataFrame has undergone some
        transformations or preprocessing steps before being passed to the "data_check" method
        """
        self.check_data_types(train_df, incoming_df)
        self.check_schema(train_df, incoming_df)
        self.check_schema(processed_train_df, processed_incoming_df)

    def get_processed_feature_importance_mapping(self):
        """
        The function `get_processed_feature_importance_mapping` loads a trained random forest classifier,
        retrieves the feature importances, retrieves the feature names from a saved column transformer, and
        creates a mapping of feature names to their corresponding importances.
        :return: a dictionary called "processed_feature_importance_mapping" which contains the feature names
        as keys and their corresponding importance values as values.
        """
        RF_clf = load('./model/RF_clf.joblib')
        feature_importance = RF_clf.feature_importances_
        # print(f"feature importances : {feature_importance}, type : {type(feature_importance)}")
        column_transformer = load('./preprocessor/column_transformer.pkl')
        feature_names = column_transformer.get_feature_names_out()
        # print(f"feature names : {feature_names}, type : {type(feature_names)}")
        processed_feature_importance_mapping = {}
        for name, importance in zip(feature_names, feature_importance):
            processed_feature_importance_mapping.update({name : importance})

        print(f"processed feature importance mapping : {processed_feature_importance_mapping}")
        return processed_feature_importance_mapping
    
    def get_feature_importance_mapping(self, processed_feature_importance_mapping):
        """
        The function `get_feature_importance_mapping` calculates the total importance of features and
        creates a mapping of feature names to their respective importance values.
        
        :param processed_feature_importance_mapping: The parameter `processed_feature_importance_mapping` is
        a dictionary that contains the feature importance values for each feature and their respective unique
        values for categorical features. The keys of the dictionary represent the features, and the values 
        represent their importance
        :return: a tuple containing the feature_importance_mapping dictionary and the total_importance
        value.
        """
        columns_list = self.columns_list
        total_importance = 0.00
        feature_importance_mapping = {}
        for value in processed_feature_importance_mapping.values():
            total_importance += value

        for col in columns_list:
            importance = 0.00
            for key, value in processed_feature_importance_mapping.items():
                if col in key:
                    importance += value
                else:
                    continue
            feature_importance_mapping.update({col : importance})
        print(total_importance)
        print(feature_importance_mapping)
        return feature_importance_mapping, total_importance
    
    def check_dataset_drift(self):
        """
        The function `check_dataset_drift` checks for dataset drift by comparing feature importance scores
        and identifying columns that have drifted.
        """
        processed_feature_importance_mapping = self.get_processed_feature_importance_mapping()
        feature_importance_mapping, total_importance = self.get_feature_importance_mapping(processed_feature_importance_mapping)
        columns_drifted = []
        with open('../json_reports/feature_drift_report.json', 'r') as f:
            data = json.load(f)
        columns_list = self.columns_list
        importance_score_drifted = 0.00
        for col in columns_list:
            if (data['metrics'][0]['result']['drift_by_columns'][col]['drift_detected'] == True):
                importance_score_drifted += feature_importance_mapping[col]
                columns_drifted.append(col)
            else:
                continue
        dataset_drift_percentage = importance_score_drifted / total_importance
        if dataset_drift_percentage > 0.5:
            print(f"dataset has drifted")
            print(f"dataset drift percentage is : {dataset_drift_percentage}")
            print(f"these are the columns that have drifted {columns_drifted}")
        else:
            print(f"dataset drift is not significant")
            print(f"dataset drift percentage is : {dataset_drift_percentage}")
            print(f"these are the columns that have drifted {columns_drifted}")