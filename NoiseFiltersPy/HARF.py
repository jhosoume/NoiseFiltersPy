from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import typing as t

from NoiseFiltersPy.Filter import *


class HARF:
    def __init__(self, nfolds: int = 10, agreementLevel: float = 0.7,
                 ntrees: int = 500, seed: int = 0, n_jobs: int = -1):
        # Some data verification
        # Data can be a DataFrame or a Numpy Array
        if (agreementLevel < 0.5 or agreementLevel > 1):
            raise ValueError("Agreement Level must be between 0.5 and 1.")
        # if (classColumn < 0 or classColumn > len(data)):
        #     raise ValueError("Column of class out of data bounds")
        self.nfolds = nfolds
        self.agreementLevel = agreementLevel
        self.ntrees = ntrees
        self.seed = seed
        self.n_jobs = n_jobs
        self.k_fold = KFold(nfolds, shuffle = True, random_state = self.seed)
        self.clf = RandomForestClassifier(n_estimators = ntrees, random_state = seed, n_jobs = self.n_jobs)
        self.filter = Filter(parameters = {"nfolds": self.nfolds, "ntrees": self.ntrees, "agreementLevel": self.agreementLevel})


    def __call__(self, data: t.Sequence, classes: t.Sequence) -> Filter:
        self.splits = self.k_fold.split(data)
        self.isNoise = np.array([False] * len(classes))
        for train_indx, test_indx in self.splits:
            self.clf.fit(data[train_indx], classes[train_indx])
            probs = self.clf.predict_proba(data[test_indx])
            self.isNoise[test_indx] = [prob[class_indx] <= 1 - self.agreementLevel
                                       for prob, class_indx in zip(probs, classes[test_indx])]
        self.filter.rem_indx = np.argwhere(self.isNoise)
        notNoise = np.invert(self.isNoise)
        self.filter.set_cleanData(data[notNoise], classes[notNoise])
        return self.filter
