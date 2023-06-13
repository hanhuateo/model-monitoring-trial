from monitoring.evidently import *
import pandas as pd
from evidently import ColumnMapping

X_combined = pd.read_csv('../data/split_data/X_combined.csv')
X_current = pd.read_csv('../data/split_data/X_current.csv')

X_combined_processed = pd.read_csv('../data/processed/X_combined_processed.csv')
X_current_processed = pd.read_csv('../data/processed/X_current_processed.csv')

y_combined = pd.read_csv('../data/split_data/y_combined.csv')
y_current = pd.read_csv('../data/split_data/y_current.csv')

y_combined_processed = pd.read_csv('../data/processed/y_combined_processed.csv')
y_current_processed = pd.read_csv('../data/processed/y_current_processed.csv')

y_prediction = pd.read_csv('../data/predicted/y_prediction.csv')

column_mapping = ColumnMapping()
column_mapping.target = 'Attrition'
column_mapping.prediction = 'Prediction'
column_mapping.id = None
column_mapping.categorical_features = ['Department',
                                       'EducationField',
                                       'Gender',
                                       'JobRole',
                                       'MaritalStatus',
                                       'OverTime',
                                       'BusinessTravel',
                                       'Education',
                                       'EnvironmentSatisfaction',
                                       'JobInvolvement',
                                       'JobLevel',
                                       'JobSatisfaction',
                                       'PerformanceRating',
                                       'RelationshipSatisfaction',
                                       'StockOptionLevel',
                                       'WorkLifeBalance']
column_mapping.numerical_features = ['Age', 
                                     'DailyRate', 
                                     'DistanceFromHome', 
                                     'HourlyRate',
                                     'MonthlyIncome',   
                                     'MonthlyRate',
                                     'NumCompaniesWorked',
                                     'PercentSalaryHike',
                                     'TotalWorkingYears',
                                     'TrainingTimesLastYear',
                                     'YearsAtCompany',
                                     'YearsInCurrentRole',
                                     'YearsSinceLastPromotion',
                                     'YearsWithCurrManager']

data_drift_report = Report(metrics=[
    # DataDriftPreset(cat_stattest='jensenshannon', num_stattest='ks'),
    # DataDriftPreset(cat_stattest='jensenshannon', num_stattest='jensenshannon')
    DataDriftPreset(),
])

data_drift_report.run(reference_data=X_combined, current_data=X_current)
data_drift_report.save_html('./monitoring/reports/data_drift_report.html')

data_drift_test_suite = TestSuite(tests=[
    DataDriftTestPreset(),
])

data_drift_test_suite.run(reference_data=X_combined, current_data=X_current)
data_drift_test_suite.save_html('./monitoring/reports/data_drift_test_suite.html')

data_quality_report = Report(metrics=[
    DataQualityPreset(),
])

data_quality_report.run(reference_data=X_combined, current_data=X_current)
data_quality_report.save_html('./monitoring/reports/data_quality_report.html')

data_quality_test_suite = TestSuite(tests=[
    DataQualityTestPreset(),
])

data_quality_test_suite.run(reference_data=X_combined, current_data=X_current)
data_quality_test_suite.save_html('./monitoring/reports/data_quality_test_suite.html')

num_target_drift_report = Report(metrics=[
    TargetDriftPreset(),
])
num_target_drift_report.run(reference_data=y_combined, current_data=y_current)
num_target_drift_report.save_html('./monitoring/reports/num_target_drift_report.html')