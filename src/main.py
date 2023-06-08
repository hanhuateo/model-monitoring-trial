import numpy as pandas
import seaborn as sns
import pandas as pd
from joblib import load
from monitoring.evidently import *
from data_manipulation.cleaning import *
from data_manipulation.understandings import *
from data_manipulation.preprocess import *

RF_clf = load('../data/models/RF_clf.joblib')
reference = pd.read_csv('../data/split_data/train_dataset.csv')
current = pd.read_csv('../data/split_data/test_dataset.csv').drop(columns='Attrition', inplace=True)

# see if got drift between reference and current data
# also run the current data through the random forest classifier and compare the
# evaluation curves and performance metrics

report = Report(metrics=[])

test = TestSuite(tests=[])
test = setNoTargetTest(test)
reference['Education'].astype(int)
test.run(current_data=current, reference_data=reference)
test.save_html('./monitoring/reports/NoTargetTestsReport.html')
