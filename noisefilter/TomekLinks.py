import numpy as np

from noisefilter.Filter import *


class TomekLinks:
    def __init__(self):
        self.filter = Filter(parameters = {})

    def __call__(self, data, classes):
        levels = list(set(classes))
        classes = np.array(classes)
        class1Indxes = np.argwhere(classes == levels[0])
        class2Indxes = np.argwhere(classes == levels[1])
        tomekMatrix = np.ones((len(class1Indxes), len(class2Indxes)), dtype = bool)
        for row, c1indx in enumerate(class1Indxes):
            for column, c2indx in enumerate(class2Indxes):
                meanPoint = (data[c1indx] + data[c2indx]) / 2
                except1indx = np.delete(class1Indxes, row)
                dist1 = np.sum(np.abs(data[except1indx] - meanPoint), axis = 1)
                if np.any(dist1 <= np.sum(np.abs(data[c1indx] - meanPoint))):
                    tomekMatrix[row, column] = False
                except2indx = np.delete(class2Indxes, column)
                dist2 = np.sum(np.abs(data[except2indx] - meanPoint), axis = 1)
                if np.any(dist2 <= np.sum(np.abs(data[c2indx] - meanPoint))):
                    tomekMatrix[row, column] = False
        c1remove = class1Indxes[np.sum(tomekMatrix, axis = 1) > 0]
        c2remove = class2Indxes[np.sum(tomekMatrix, axis = 0) > 0]
        toRemove = np.concatenate((c1remove, c2remove))

        self.filter.remIndx = np.sort(toRemove)
        self.filter.remIndx.sort()
        self.filter.set_cleanData(np.delete(data, self.filter.remIndx, axis = 0),
                                  np.delete(classes, self.filter.remIndx, axis = 0))
        return self.filter
