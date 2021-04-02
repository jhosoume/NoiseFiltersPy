from sklearn.utils.validation import check_random_state
import numpy as np
from NoiseFiltersPy.Injector import Injector

class RandomInjector(Injector):
    def generate(self, seed: int = None):
        rng = np.random.default_rng(seed)
        self._new_noise = rng.choice(self._labels.shape[0], size = self._num_noise, replace = False)
        self._gen_random(seed = seed)
        return self