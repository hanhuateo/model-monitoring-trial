import numpy as np
from scipy import stats
from evidently.metrics import DataDriftTable
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
import json
import win32com.client

class ModelMonitoring():
    def __init__(self, train_df):
        """
        The function initializes the object with the numerical and categorical columns of a given
        dataframe, as well as empty dictionaries for statistical tests and their thresholds.
        
        :param train_df: The train_df parameter is a pandas DataFrame that contains the training data
        for your machine learning model. It is used to initialize the object and extract the numerical
        and categorical columns from the DataFrame
        """
        self.columns_list = train_df.columns.tolist()
        self.numerical_columns = self.get_numerical_columns(train_df)
        self.categorical_columns = self.get_categorical_columns(train_df)
        self.stat_test_foreach_column = {}
        self.stat_test_threshold_foreach_column = {}
    
    def get_numerical_columns(self, df):
        """
        The function "get_numerical_columns" takes a dataframe as input and returns a list of column
        names that contain numerical data.
        
        :param df: The parameter "df" is a pandas DataFrame object
        :return: a list of column names that contain numerical data in the given dataframe.
        """
        list_column_names = list(df.columns)
        numerical_list = []
        for col in list_column_names:
            if ((df[col].dtype in [np.int64,float])):
                numerical_list.append(col)
                continue
        return numerical_list

    def get_categorical_columns(self, df):
        """
        The function "get_categorical_columns" takes a dataframe as input and returns a list of column
        names that contain categorical data.
        
        :param df: The parameter "df" is a pandas DataFrame object
        :return: a list of column names that have categorical data types in the given dataframe.
        """
        list_column_names = list(df.columns)
        categorical_list = []
        for col in list_column_names:
            if ((df[col].dtype in [object, str, 'category'])):
                categorical_list.append(col)
        return categorical_list
    
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
    
    def set_stat_test_foreach_column(self, test_df, customise = False):
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
        with open("config.json", "r") as jsonfile:
            data = json.load(jsonfile)
        print(f"categorical threshold is : {data['categorical_threshold']}")
        print(f"numerical threshold is : {data['numerical_threshold']}")
        customise = customise
        print(f"customise in set_stat_test_foreach_column : {customise}")
        if (customise == 'False'):
            numerical_column_dictionary = {}
            categorical_column_dictionary = {}
            numerical_column_dictionary = self.numerical_stat_test_algo(test_df, self.numerical_columns, len(test_df), numerical_column_dictionary)
            categorical_column_dictionary = self.categorical_stat_test_algo(test_df, self.categorical_columns, len(test_df), categorical_column_dictionary)
            self.stat_test_foreach_column = {**categorical_column_dictionary, **numerical_column_dictionary}
            for col in self.categorical_columns:
                self.stat_test_threshold_foreach_column.update({col:data['categorical_threshold']})
            for col in self.numerical_columns:
                self.stat_test_threshold_foreach_column.update({col:data['numerical_threshold']})
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
                self.stat_test_threshold_foreach_column.update({col : stat_test_threshold})
    
    def feature_drift_report(self, train_df, test_df, format):
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
        report will be saved as an HTML file. If 'json' is chosen, the report
        :return: the feature drift report as a dictionary.
        """
        print("do you want to customise the stat test for each column?")
        print("if yes, input True, else input False")
        customise = input("Choice:")
        print(f"customise in feature_drift_report : {customise}")
        self.set_stat_test_foreach_column(test_df, customise)
        feature_drift_report = Report(metrics = [
            DataDriftTable(per_column_stattest=self.stat_test_foreach_column,
                           per_column_stattest_threshold=self.stat_test_threshold_foreach_column),
        ])
        feature_drift_report.run(reference_data=train_df, current_data=test_df)
        if format == 'html':
            feature_drift_report.save_html('../html_reports/feature_drift_report.html')
        else:
            feature_drift_report.save_json('../json_reports/feature_drift_report.json')
        return feature_drift_report.as_dict()

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
        print("The available stats tests are: \n")
        print("For categorical: ")
        print("chisquare, z, fisher_exact, g_test, TVD")
        print("For numerical: ")
        print("ks, wasserstein, anderson, cramer_von_mises, mannw, ed, es, t_test, emperical_mmd")
        print("For both categorical and numerical: ")
        print("kl_div, psi, jensenshannon, hellinger")
        stat_test = input(f"What is your stat test of choice?")
        stat_test_threshold = input(f"What is the threshold for your stat test?")
        stat_test_threshold = float(stat_test_threshold)
        print(stat_test_threshold)
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
    
    def check_for_drift(self, option):
        columns_list = self.columns_list
        feature_drift_list = []
        if option == 'feature':
            f = open('../json_reports/feature_drift_report.json')
            json_report = json.load(f)
            for col in columns_list:
                if (json_report['metrics'][0]['result']['drift_by_columns'][col]['drift_score'] < json_report['metrics'][0]['result']['drift_by_columns'][col]['stattest_threshold']):
                    feature_drift_list.append(col)

            if bool(feature_drift_list):    
                ol=win32com.client.Dispatch("outlook.application")
                olmailitem=0x0 #size of the new email
                newmail=ol.CreateItem(olmailitem)
                newmail.Subject= 'Feature Drift'
                newmail.To='hanhuateo@gmail.com'
                # newmail.CC='xyz@gmail.com'
                newmail.Body="Hello, these are the columns that have drifted : " + feature_drift_list
                newmail.Send()
            f.close()

        if option == 'prediction':
            f = open('../json_reports/prediction_drift_report.json')
            json_report = json.load(f)
            if (json_report['metrics'][0]['result']['drift_score'] < json_report['metrics'][0]['result']['stattest_threshold']):
                ol = win32com.client
                olmailitem=0x0
                newmail = ol.CreateItem(olmailitem)
                newmail.Subject = 'Prediction Drift'
                newmail.To='hanhuateo@gmail.com'
                # newmail.CC='xyz@gmail.com'
                newmail.Body = "Hello, prediction column has drifted"
                newmail.Send()

    def check_schema(self, train_df, test_df):
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
        """
        The function checks if the processed train and test datasets have the same features and raises
        an exception if they don't.
        
        :param processed_train_df: The processed_train_df parameter is a DataFrame that represents the
        processed training dataset. It contains the features (columns) that have been preprocessed and
        transformed for training the model
        :param processed_test_df: The parameter `processed_test_df` is a DataFrame that represents the
        processed test dataset. It contains the features and corresponding values for each instance in
        the test dataset
        :return: 1 if the processed_train_df and processed_test_df have the same processed features. If
        they do not have the same processed features, the function raises an exception.
        """
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
            if (train_df[col].dtype == test_df[col].dtype):
                # print(f"Data Type of {col} from both datasets are the same")
                continue
            else:
                print(f"Column name is : {col}")
                raise TypeError(f"Data Type of {col} in production does not match with training")
        print(f"The data types for all columns from both datasets are the same")

    def replace_column_names(self, df):   
        """
        The function replaces spaces and dashes in column names of a DataFrame with underscores.
        
        :param df: The parameter `df` is a pandas DataFrame object
        """
        df.rename(columns=lambda s: s.replace(" ", "_" ), inplace=True)
        df.rename(columns=lambda s: s.replace("-", "_"), inplace=True) 

    def handle_bad_data(self, df):
        """
        The function replaces any occurrences of '?' or '-' in a DataFrame with NaN values.
        
        :param df: The parameter `df` is a pandas DataFrame object
        :return: the modified dataframe with '?' and '-' values replaced with NaN.
        """
        df = df.replace(['?', '-'], np.nan)
        return df

    def data_check(self, train_df, test_df, processed_train_df, processed_test_df):
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