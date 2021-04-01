import numpy as np
import pandas as pd
import gower
from NoiseFiltersPy.Injector import Injector

class NeighborwiseInjector(Injector):
    def generate(self, seed: int = None):
        self._new_noise = self._define_noise_examples()
        self._gen_random(seed = seed)

    def _cal_dNN(self, distances, example_indx):
        equal_class = self._labels == self._labels.iloc[example_indx]
        unequal_class = ~equal_class
        equal_class.iloc[example_indx] = False
        intra_class_distance = min(distances[example_indx, np.ravel(equal_class)])
        inter_class_distance = min(distances[example_indx, np.ravel(unequal_class)])
        # TODO Check division by zero
        return intra_class_distance/inter_class_distance
    
    def _define_noise_examples(self):
        distances = gower.gower_matrix(self._attrs)
        min_distances = [self._cal_dNN(distances, indx) for indx in range(self._labels.shape[0])]
        min_distances = pd.DataFrame(min_distances, columns = ["distance"])
        min_distances.sort_values("distance", ascending = False, inplace = True)
        return list(min_distances[:self._num_noise].index)
