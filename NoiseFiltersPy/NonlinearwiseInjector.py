import itertools
import numpy as np
import pandas as pd
from sklearn import svm
from NoiseFiltersPy.Injector import Injector

class NonlinearwiseInjector(Injector):
    def generate(self, seed: int = None):
        self._new_noise = self._define_noise_examples(seed = seed)
        self._gen_random(seed = seed)

    def _one_vs_one(self):
        labels_comb = itertools.combinations(self._label_types, 2)
        bins = []
        for comb in labels_comb:
            in_comb = self._labels.isin(comb)
            bins.append({
                "attrs": self._attrs[in_comb],
                "labels": self._labels[in_comb]
            })
        return bins
    
    def _svm(self, bin, seed: int = None):
        clf = svm.SVC(kernel = "rbf", gamma = "auto", random_state = seed)
        clf.fit(bin["attrs"], bin["labels"])
        return clf
    
    def _define_noise_examples(self, seed: int = None):
        clf = svm.SVC(
            kernel = "rbf", 
            gamma = "auto",
            decision_function_shape = "ovo",
            random_state = seed
        )
        clf.fit(self._attrs, np.ravel(self._labels))
        distances = clf.decision_function(self._attrs)
        if distances.ndim > 1:
            # When multiple attributes, get the minimum distance
            distances = pd.DataFrame(np.apply_along_axis(
                min,
                axis = 1,
                arr = abs(distances)
            ), columns = ["distance"])
        else:
            distances = pd.DataFrame(abs(distances), columns = ["distance"])
        return list(distances.sort_values("distance", ascending = True)[:self._num_noise].index)
        
        
