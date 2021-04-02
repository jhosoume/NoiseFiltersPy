import pandas as pd
from NoiseFiltersPy._filters import _implemented_filters
from NoiseFiltersPy._injectors import _implemented_injectors
from NoiseFiltersPy._helpers import timeit

class NoiseHandler:
    """Core class for noise injection or removal. 
    """
    def __init__(self, measure_time: bool = False, out_type = pd.DataFrame) -> None:
        """Integrates injectors and filters, grouping all the functions.

        Fit must be called before filtering or injection.

        Args:
            measure_time (bool, optional): [description]. Defaults to False.
        """
        self._measure_time = measure_time        
        self._out_type = out_type

    def fit(self, attributes, labels):
        self._attrs = attributes
        self._labels = labels

    def filter(self, methods):
        filters = []
        times = []
        for filter_type in methods:
            if filter_type in _implemented_filters:
                filter = _implemented_filters[filter_type]()
                if self._measure_time:
                    filter, time = timeit(
                        filter,
                        data = self._attrs,
                        classes = self._labels 
                    )
                    times.append(time)
                else:
                    filter = filter(
                        filter,
                        data = self._attrs,
                        classes = self._labels 
                    )
                self._attrs = filter.clean_data
                self._classes = filter.clean_classes
                filters.append(filter)
        if self._measure_time:
            return filters, times
        return filters


    def inject(self, methods):
        injectors = []
        times = []
        for injector_type in methods:
            if injector_type in _implemented_injectors:
                injector = _implemented_injectors[injector_type]()
                if self._measure_time:
                    filter, time = timeit(
                        injector,
                        data = self._attrs,
                        classes = self._labels 
                    )
                    times.append(time)
                else:
                    filter = filter(
                        filter,
                        data = self._attrs,
                        classes = self._labels 
                    )
                injectors.append(filter)
        if self._measure_time:
            return injectors, times
        return injectors 

    def _output_converter(self):
        raise NotImplementedError