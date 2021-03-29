import typing as t

class Filter:
    """Base class for all the implemented class noise filters.

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

    def __init__(self, parameters: t.Dict):
        """

        Parameters
        ----------
        parameters : :obj:`Dict`
            Dictionary that provides hyperparameters for filters algorithms.
        """
        # Removed Indexes
        self.rem_indx: t.List = []
        self.parameters = parameters

    def set_cleanData(self, data: t.Sequence, classes: t.Sequence) -> t.NoReturn:
        """Helper function to set data and classes to Filter instance.

        Parameters
        ----------
        data : :obj:`Sequence`
            Filtered independent attributes(X) of the dataset.
        classes : :obj:`Sequence`
            Filtered target attributes(y) of the dataset.

        """
        self.clean_data = data
        self.clean_classes = classes
