class Filter():
    def __init__(self, parameters):
        # Removed Indexes
        self.remIndx = []
        self.repIndx = []
        self.parameters = parameters
    def set_cleanData(self, data, classes):
        self.cleanData = data
        self.cleanClasses = classes
