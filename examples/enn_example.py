from sklearn import datasets
from NoiseFiltersPy.ENN import ENN

dataset = datasets.load_iris()
data = dataset.data
classes = dataset.target
enn = ENN()
filter = enn(data, classes)
