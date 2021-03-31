import itertools
import numpy as np
import pandas as pd
import sklearn
from NoiseFiltersPy.Injector import Injector

class NonlinearwiseInjector(Injector):
    def generate(self, seed: int = None):
        self._define_noise_examples(seed = seed)
        self._gen_random(seed = seed)

    def _one_vs_one(self):
        labels_comb = itertools.combinations(self.label_types, 2)
        bins = []
        for comb in labels_comb:
            in_comb = self.labes.isin(comb)
            bins.append({
                "attrs": self.attrs[in_comb],
                "labels": self.labels[in_comb]
            })
        return bins
    
    def _svm(self, bin, seed: int = None):
        clf = sklearn.svm.SVC(kernel = "rbf", gamma = "auto", random_state = seed)
        clf.fit(bin["attrs"], bin["labels"])
        return clf
    
    def _define_noise_examples(self, seed: int = None):
        clf = sklearn.svm.SVC(
            kernel = "rbf", 
            gamma = "auto",
            decision_function_shape = "ovo",
            random_state = seed
        )
        clf.train(self.attrs, np.ravel(self.classes))
        distances = clf.decision_function(self.attrs)
        distances = pd.DataFrame(np.apply_along_axis(
            min,
            axis = 1,
            arr = abs(distances)
        ), columns = ["distance"])
        return list(distances.sort_values("distance", ascending = False)[:self.num_noise].index)
        
        
