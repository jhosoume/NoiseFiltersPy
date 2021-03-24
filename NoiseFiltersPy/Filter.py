import typing as t

class Filter():
    def __init__(self, parameters: t.Dict):
        # Removed Indexes
        self.remIndx: t.List = []
        self.parameters = parameters

    def set_cleanData(self, data: t.Sequence, classes: t.Sequence) -> t.NoReturn:
        # Helper function to set data without noise
        self.cleanData = data
        self.cleanClasses = classes
