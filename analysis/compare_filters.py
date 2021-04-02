from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from scipy.io import arff as arff_io
from sklearn import preprocessing, metrics

from NoiseFiltersPy._filters import _implemented_filters
from NoiseFiltersPy._injectors import _implemented_injectors 

DATASETS_PATH = "analysis/datasets/"

datasets = [f for f in listdir(DATASETS_PATH)
                if ( isfile(join(DATASETS_PATH, f)) and
                ( f.endswith("json") or f.endswith("arff") ) )]

enc = preprocessing.OneHotEncoder(handle_unknown = 'ignore')
le = preprocessing.LabelEncoder()

def calculate_filter_f1(dataset, filter, injector, rate = 0.1):
    # Reading dataset
    if dataset.endswith("json"):
        data = pd.read_json(DATASETS_PATH + dataset)
    elif dataset.endswith("arff"):
        data = arff_io.loadarff(DATASETS_PATH + dataset)
        data = pd.DataFrame(data[0])
    target = data["class"].values
    # Data preprocessing (type transformation)
    if target.dtype == object:
        le.fit(target)
        target = le.transform(target)
    attrs = data.drop("class", axis = 1).values
    if not np.issubdtype(attrs.dtype, np.number):
        enc.fit(attrs)
        attrs = enc.transform(attrs).toarray()

    injector = injector(attrs, target, rate)
    injector.generate()

    filter = filter()
    filter = filter(attrs, np.ravel(injector.labels.values))
    real_values = [1 if indx in injector.noise_indx else 0 for indx in range(len(target))]
    pred_values = [1 if indx in filter.rem_indx else 0 for indx in range(len(target))]
    return metrics.f1_score(real_values, pred_values, average = "micro")

results = {}
for filter in _implemented_filters.keys():
    results[filter] = {}
    for injector in _implemented_injectors.keys():
        results[filter][injector] = {}
        for dataset in datasets:
            results[filter][injector][dataset] = calculate_filter_f1(
                dataset,
                _implemented_filters[filter], 
                _implemented_injectors[injector]
            )

results = pd.DataFrame(results)
results.to_csv("compare_filters.csv")