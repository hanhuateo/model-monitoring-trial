from monitoring.evidently import *
import pandas as pd

X_combined = pd.read_csv('../data/split_data/X_combined.csv')
X_current = pd.read_csv('../data/split_data/X_current.csv')

X_combined_processed = pd.read_csv('../data/processed/X_combined_processed.csv')
X_current_processed = pd.read_csv('../data/processed/X_current_processed.csv')

y_combined = pd.read_csv('../data/split_data/y_combined.csv')
y_current = pd.read_csv('../data/split_data/y_current.csv')

y_combined_processed = pd.read_csv('../data/processed/y_combined_processed.csv')
y_current_processed = pd.read_csv('../data/processed/y_current_processed.csv')

data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(reference_data=X_combined, current_data=X_current)
data_drift_report.save_html('./monitoring/reports/data_drift_report.html')