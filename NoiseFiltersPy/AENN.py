from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import typing as t

from NoiseFiltersPy.Filter import *


class AENN:
    def __init__(self, max_neighbours: int = 5):
        self.max_neighbours = max_neighbours
        self.filter = Filter(parameters = {"max_neighbours": self.max_neighbours})

    def __call__(self, data: t.Sequence, classes: t.Sequence) -> Filter:
        self.isNoise = np.array([False] * len(classes))
        for n_neigh in range(1, self.max_neighbours + 1):
            self.clf = KNeighborsClassifier(n_neighbors = n_neigh, algorithm = 'kd_tree', n_jobs = -1)
            for indx in np.argwhere(np.invert(self.isNoise)):
                self.clf.fit(np.delete(data, indx, axis = 0), np.delete(classes, indx, axis = 0))
                pred = self.clf.predict(data[indx])
                self.isNoise[indx] = pred != classes[indx]
        self.filter.remIndx = np.argwhere(self.isNoise)
        notNoise = np.invert(self.isNoise)
        self.filter.set_cleanData(data[notNoise], classes[notNoise])
        return self.filter
