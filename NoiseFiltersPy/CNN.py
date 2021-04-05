from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from NoiseFiltersPy.Filter import *


class CNN:
    def __init__(self, max_neighbours: int = 5, n_jobs: int = -1):
        self.max_neighbours = max_neighbours
        self.filter = Filter(parameters = {})
        self.n_jobs = n_jobs
        self.clf = KNeighborsClassifier(n_neighbors = 1, n_jobs = self.n_jobs)

    def __call__(self, data: t.Sequence, classes: t.Sequence):
        self.isNoise = np.array([False] * len(classes))

        firstDifIndx = next(indx for indx, num in enumerate(classes) if num != classes[0])
        inStore = [0, firstDifIndx]
        grabBag = [indx for indx in range(1, firstDifIndx)]
        for indx in range(firstDifIndx + 1, len(classes)):
            self.clf.fit(data[inStore], classes[inStore])
            pred = self.clf.predict(data[indx].reshape(1, -1))
            if pred == classes[indx]:
                grabBag.append(indx)
            else:
                inStore.append(indx)
        keepOn = True
        while(keepOn):
            keepOn = False
            for indx in grabBag:
                self.clf.fit(data[inStore], classes[inStore])
                pred = self.clf.predict(data[indx].reshape(1, -1))
                if (pred != classes[indx]):
                    inStore.append(indx)
                    grabBag.remove(indx)
                    keepOn = True
        self.filter.rem_indx = grabBag
        self.filter.rem_indx.sort()
        notNoise = inStore
        notNoise.sort()
        self.filter.set_cleanData(data[notNoise], classes[notNoise])
        return self.filter
