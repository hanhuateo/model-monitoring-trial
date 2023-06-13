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

column_mapping = ColumnMapping()
column_mapping.target = 'Attrition'


data_drift_report = Report(metrics=[
    # DataDriftPreset(cat_stattest='jensenshannon', num_stattest='ks'),
    # DataDriftPreset(cat_stattest='jensenshannon', num_stattest='jensenshannon')
    DataDriftPreset(),
])

data_drift_report.run(reference_data=X_combined, current_data=X_combined_processed)
data_drift_report.save_html('./monitoring/reports/data_drift_report.html')

data_drift_test_suite = TestSuite(tests=[
    DataDriftTestPreset(),
])

data_drift_test_suite.run(reference_data=X_combined, current_data=X_current)
data_drift_test_suite.save_html('./monitoring/reports/data_drift_test_suite.html')