from sklearn import datasets
from NoiseFiltersPy.TomekLinks import TomekLinks

dataset = datasets.load_iris()
data = dataset.data
classes = dataset.target
tomek = TomekLinks()
filter = tomek(data, classes)
