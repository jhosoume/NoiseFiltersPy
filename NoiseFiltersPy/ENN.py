from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from NoiseFiltersPy.Filter import *

class ENN:
    def __init__(self, neighbours = 3):
        self.neighbours = neighbours
        self.filter = Filter(parameters = {"neighbours": self.neighbours})

    def __call__(self, data, classes):
        self.isNoise = np.array([False] * len(classes))
        self.clf = KNeighborsClassifier(n_neighbors = self.neighbours, algorithm = 'kd_tree', n_jobs = -1)
        for indx in range(len(data)):
            self.clf.fit(np.delete(data, indx, axis = 0), np.delete(classes, indx, axis = 0))
            pred = self.clf.predict(data[indx].reshape(1, -1))
            self.isNoise[indx] = pred != classes[indx]
        self.filter.remIndx = np.argwhere(self.isNoise)
        notNoise = np.invert(self.isNoise)
        self.filter.set_cleanData(data[notNoise], classes[notNoise])
        return self.filter
