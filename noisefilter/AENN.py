from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from noisefilter.Filter import *


class AENN:
    def __init__(self, max_neighbours = 5):
        self.max_neighbours = max_neighbours

    def __call__(self, data, classes):
        self.isNoise = np.array([False] * len(classes))
        filter = Filter(parameters = {"max_neighbours": self.max_neighbours})
        for n_neigh in range(1, self.max_neighbours):
            self.clf = KNeighborsClassifier(n_neighbors = n_neigh)
            for indx in np.argwhere(np.invert(self.isNoise)):
                self.clf.fit(np.delete(data, indx, axis = 0), np.delete(classes, indx, axis = 0))
                pred = self.clf.predict(data[indx])
                self.isNoise[indx] = pred != classes[indx]
        filter.remIndx = np.argwhere(self.isNoise)
        notNoise = np.invert(self.isNoise)
        filter.set_cleanData(data[notNoise], classes[notNoise])
        return filter
