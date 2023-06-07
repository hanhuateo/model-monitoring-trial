import numpy as pandas
import seaborn as sns
import pandas as pd
from joblib import load
from monitoring.evidently import *

RF_clf = load('../data/models/RF_clf.joblib')
reference = pd.read_csv('../data/split_data/train_dataset.csv')
current = pd.read_csv('../data/split_data/test_dataset.csv')

# see if got drift between reference and current data
# also run the current data through the random forest classifier and compare the
# evaluation curves and performance metrics

# report = Report(metrics=[])
# report = setDataDriftPreset(report)
# report.run(current_data=current, reference_data=reference)
# report.save_html('./monitoring/reports/DataDriftReport.html')

# colName1 = 'DailyRate'
# report = setColumnMetric(report, colName1)
# report.run(current_data=current, reference_data=reference)
# report.save_html('./monitoring/reports/ColumnMetricReport_{}.html'.format(colName1))
# print(report.json)

# colName2 = 'Department'
# report = setColumnMetric(report, colName2)
# report.run(current_data=current, reference_data=reference)
# report.save_html('./monitoring/reports/ColumnMetricReport_{}.html'.format(colName2))

# report = setTests(report)
# report.run(current_data=current, reference_data=reference)
# report.save_html('./monitoring/reports/TestsReport.html')
# print(report.json)

# report = setNoTargetTest(report)
# report.run(current_data=current, reference_data=reference)
# report.save_html('./monitoring/reports/NoTargetTestsReport.html')
# print(report.json)
