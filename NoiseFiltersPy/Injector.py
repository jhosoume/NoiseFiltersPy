import numpy as np
import pandas as pd
from abc import ABC


class Injector(ABC):
    """Base class for the injectors of artificial noise. 

    Attributes
    ----------
    rem_indx : :obj:`List`
        Removed indexes (rows) from the dataset after the filtering.
    parameters : :obj:`Dict`
        Parameters used to define the behaviour of the filter.
    clean_data : :obj:`Sequence`
        Filtered independent attributes(X) of the dataset.
    clean_classes : :obj:`Sequence`
        Filtered target attributes(y) of the dataset.

    """

    def __init__(self, attributes, labels, rate: float = 0.1) -> None:
        self._new_noise = []
        if not isinstance(attributes, pd.DataFrame):
            self._attrs = pd.DataFrame(attributes)
        else:
            self._attrs = attributes

        if not isinstance(labels, pd.DataFrame):
            self._labels = pd.DataFrame(labels)
        else:
            self._labels = labels

        self._rate = rate
        self.verify()
        self._num_noise = int(self._rate * self._attrs.shape[0])
        self._label_types = set(self.labels[0].unique())
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def noise_indx(self):
        return self._new_noise
    
    def verify(self) -> None:
        if min(self._labels.value_counts()) < 2:
            raise ValueError("Number of examples in the minority class must be >= 2.")
        
        if self._attrs.shape[0] != self.labels.shape[0]:
            raise ValueError("Attributes and classes must have the sime size.")

        if self._rate < 0 or self._rate > 1:
           raise ValueError("") 
    
    def _gen_random(self, seed: int = None):
        """[summary]

        Args:
            seed (int, optional): [description]. Defaults to 123.
        """
        rng = np.random.default_rng(seed)
        for example in self._new_noise:
            self._labels.iloc[example] = rng.choice(list(self._label_types - set(self._labels.iloc[example])))

    