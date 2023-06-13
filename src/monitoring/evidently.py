from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset, DataDriftTestPreset, DataQualityTestPreset
from evidently.tests import *

def setDataDriftPreset(report):
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    return report

def setColumnMetric(report, colName):
    report = Report(metrics=[
        ColumnSummaryMetric(column_name=colName),
        ColumnQuantileMetric(column_name=colName, quantile=0.25),
        ColumnDriftMetric(column_name=colName)
    ])
    return report

def setTests(tests):
    tests = TestSuite(tests=[
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
        TestNumberOfDriftedColumns(),
    ])
    return tests

def setNoTargetTest(suite):
    suite = TestSuite(tests=[
        NoTargetPerformanceTestPreset(),
    ])
    return suite