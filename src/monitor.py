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

train_dataset = pd.read_csv('../data/split_data/train_dataset.csv')

column_mapping = ColumnMapping()
column_mapping.target = 'Attrition'
column_mapping.prediction = 'Prediction'
column_mapping.id = None
column_mapping.task = 'classification'
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
    # DataDriftPreset(),
    DatasetDriftMetric(),
    DataDriftTable(num_stattest='ks', num_stattest_threshold=0.05, cat_stattest='chisquare', cat_stattest_threshold=0.05),
    ColumnDriftMetric(column_name='Age'),
    # TextDescriptorsDriftMetric(column_name=''),
    # EmbeddingsDriftMetric(embeddings_name=''),
])
data_drift_report.run(reference_data=X_combined, current_data=X_current)
data_drift_report.save_html('./monitoring/reports/data_drift_report.html')

# data_drift_test_suite = TestSuite(tests=[
#     # DataDriftTestPreset(),
#     TestShareOfDriftedColumns(),
#     TestColumnDrift(column_name='Age'),
#     TestNumberOfDriftedColumns(),
#     # TestEmbeddingsDrift(embeddings_name=''),
# ])
# data_drift_test_suite.run(reference_data=X_combined, current_data=X_current)
# data_drift_test_suite.save_html('./monitoring/reports/data_drift_test_suite.html')
# data_drift_test_suite.save_json('./monitoring/reports/data_drift_test_suite.json')

# data_quality_report = Report(metrics=[
#     DataQualityPreset(),
# ])
# data_quality_report.run(reference_data=X_combined, current_data=X_current)
# data_quality_report.save_html('./monitoring/reports/data_quality_report.html')

# data_quality_test_suite = TestSuite(tests=[
#     DataQualityTestPreset(),
# ])
# data_quality_test_suite.run(reference_data=X_combined, current_data=X_current)
# data_quality_test_suite.save_html('./monitoring/reports/data_quality_test_suite.html')

# num_target_drift_report = Report(metrics=[
#     TargetDriftPreset(),
# ])
# num_target_drift_report.run(reference_data=train_dataset, current_data=X_current)
# num_target_drift_report.save_html('./monitoring/reports/num_target_drift_report.html')