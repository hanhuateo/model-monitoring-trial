import alibi
import matplotlib.pyplot as plt
import numpy as np

from alibi_detect.cd import ChiSquareDrift, TabularDrift
from alibi_detect.saving import save_detector, load_detector

adult = alibi.datasets.fetch_adult()
X, y = adult.data, adult.target
feature_names = adult.feature_names
category_map = adult.category_map
print(X.shape, y.shape) 

n_ref = 10000
n_test = 10000

X_ref, X_t0, X_t1 = X[:n_ref], X[n_ref:n_ref + n_test], X[n_ref + n_test:n_ref + 2 * n_test]
print(X_ref.shape, X_t0.shape, X_t1.shape)

categories_per_feature = {f: None for f in list(category_map.keys())}

cd = TabularDrift(X_ref, p_val=.05, categories_per_feature=categories_per_feature)

preds = cd.predict(X_t0)
labels = ['No!', 'Yes!']
print('Drift? {}'.format(labels[preds['data']['is_drift']]))