from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import typing as t

from NoiseFiltersPy.Filter import *

class ENN:
    def __init__(self, neighbours: int = 3, n_jobs: int = -1):
        self.neighbours = neighbours
        self.filter = Filter(parameters = {"neighbours": self.neighbours})
        self.n_jobs = n_jobs

    def __call__(self, data: t.Sequence, classes: t.Sequence) -> Filter:
        self.isNoise = np.array([False] * len(classes))
        self.clf = KNeighborsClassifier(n_neighbors = self.neighbours, algorithm = 'kd_tree', n_jobs = self.n_jobs)
        for indx in range(len(data)):
            self.clf.fit(np.delete(data, indx, axis = 0), np.delete(classes, indx, axis = 0))
            pred = self.clf.predict(data[indx].reshape(1, -1))
            self.isNoise[indx] = pred != classes[indx]
        self.filter.rem_indx = np.argwhere(self.isNoise)
        notNoise = np.invert(self.isNoise)
        self.filter.set_cleanData(data[notNoise], classes[notNoise])
        return self.filter
