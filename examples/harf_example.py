from sklearn import datasets
from noisefilter.Harf import Harf

dataset = datasets.load_iris()
data = dataset.data
classes = dataset.target
harf = Harf()
filter = harf(data, classes)
