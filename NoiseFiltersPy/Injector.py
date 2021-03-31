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
        self.new_noise = []
        if not isinstance(attributes, pd.DataFrame):
            self.attrs = pd.DataFrame(attributes)
        else:
            self.attrs = attributes

        if not isinstance(attributes, pd.DataFrame):
            self.labels = pd.DataFrame(labels)
        else:
            self.labels = labels

        self.rate = rate
        self.verify()
        self.num_noise = int(self.rate * self.attrs.shape[0])
        self.label_types = set(self.labels[0].unique())
    
    def verify(self) -> None:
        if min(self.labels.value_count()) < 2:
            raise ValueError("Number of examples in the minority class must be >= 2.")
        
        if self.attrs.shape[0] != self.labels.shape[0]:
            raise ValueError("Attributes and classes must have the sime size.")

        if self.rate < 0 or self.rate > 1:
           raise ValueError("") 
    
    def _gen_random(self, seed: int = None):
        """[summary]

        Args:
            seed (int, optional): [description]. Defaults to 123.
        """
        rng = np.random.default_rng(seed)
        for example in self.new_noise:
            self.labels.iloc[example] = rng.choice(list(self.label_types - set(self.labels.iloc[example])))

    