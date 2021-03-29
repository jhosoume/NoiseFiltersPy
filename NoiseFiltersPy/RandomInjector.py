from sklearn.utils.validation import check_random_state
import numpy as np
from NoiseFiltersPy.Injector import Injector

class RandomInjector(Injector):
    def gen(self, seed: int = None):
        rng = np.random.default_rng(seed)
        self.new_noise = rng.choice(self.labels.shape[0], size = self.num_noise, replace = False)
        for example in self.new_noise:
            self.labels.iloc[example] = rng.choice(list(self.label_types - set(self.labels.iloc[example])))