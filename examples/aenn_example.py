from sklearn import datasets
from noisefilter.AENN import AENN

dataset = datasets.load_iris()
data = dataset.data
classes = dataset.target
aenn = AENN()
filter = aenn(data, classes)
