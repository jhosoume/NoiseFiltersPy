import numpy as np
import pandas as pd
import gower
from NoiseFiltersPy.Injector import Injector

class NeighwiseInjector(Injector):
    def generate(self, seed: int = None):
        self._define_noise_examples()
        self._gen_random(seed = seed)

    def _cal_dNN(self, distances, example_indx):
        equal_class = self.classes[0] == self.classes.iloc[example, 0]
        unequal_class = ~equal_class
        equal_class.iloc[0] = False
        intra_class_distance = min(distances[example_indx, equal_class])
        inter_class_distance = min(distances[example_indx, unequal_class])
        return intra_class_distance/inter_class_distance
    
    def _define_noise_examples(self):
        distances = gower.gower_matrix(self.attrs)
        min_distances = [self._cal_dNN(distances, indx) for indx in range(self.classes.shape[0])]
        min_distances = pd.DataFrame(distances, columns = ["distance"])
        min_distances.sort_values("distance", ascending = False, inplace = True)
        self.new_noise = list(min_distances.indx[:self.num_noise].index)
