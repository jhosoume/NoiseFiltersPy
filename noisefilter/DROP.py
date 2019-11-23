from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from noisefilter.Filter import *


class DROPv1:
    def __init__(self, num_neighbours = 1):
        self.n_neigh = num_neighbours
        self.filter = Filter(parameters = {num_neighbours: self.n_neigh})

    def __call__(self, data, classes):
        self.clf = KNeighborsClassifier()
        preds = []
        for indx in range(len(classes)):
            self.clf.fit(np.delete(data, indx, axis = 0), np.delete(classes, indx, axis = 0))
            preds.append(self.clf.predict(data[indx].reshape(1, -1)))
        preds = np.array(preds)
        currentAcc = np.sum(preds.reshape(1, -1) == classes)

        indxes = np.arange(len(classes))
        toRemove = np.array([], dtype = 'int64')
        for indx in indxes:
            predsIn = []
            indxremoved = np.setdiff1d(indxes, toRemove)
            for test_indx in indxremoved:
                self.clf.fit(np.delete(data, np.concatenate(([indx], [test_indx], toRemove)), axis = 0),
                             np.delete(classes, np.concatenate(([indx], [test_indx], toRemove)), axis = 0))
                predsIn.append(self.clf.predict(data[test_indx].reshape(1, -1)))
            predsIn = np.array(predsIn)
            newAcc = np.sum(predsIn.reshape(1, -1) == classes[indxremoved])
            if (newAcc >= currentAcc):
                currentAcc = newAcc
                if predsIn[indx - len(toRemove)] == classes[indx]:
                    --currentAcc
                toRemove = np.concatenate((toRemove, [indx]))

        self.filter.remIndx = toRemove
        self.filter.remIndx = np.sort(toRemove)
        self.filter.set_cleanData(np.delete(data, self.filter.remIndx, axis = 0),
                                  np.delete(classes, self.filter.remIndx, axis = 0))
        return self.filter
