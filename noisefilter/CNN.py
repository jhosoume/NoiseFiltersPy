from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from noisefilter.Filter import *


class CNN:
    def __init__(self, max_neighbours = 5):
        self.max_neighbours = max_neighbours
        self.filter = Filter(parameters = {})
        self.clf = KNeighborsClassifier(n_neighbors = 1)

    def __call__(self, data, classes):
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
        self.filter.remIndx = grabBag
        self.filter.remIndx.sort()
        notNoise = inStore
        notNoise.sort()
        self.filter.set_cleanData(data[notNoise], classes[notNoise])
        return self.filter
