from sklearn import datasets
from noisefilter.DROP import DROPv1

dataset = datasets.load_iris()
data = dataset.data
classes = dataset.target
drop = DROPv1()
filter = drop(data, classes)
