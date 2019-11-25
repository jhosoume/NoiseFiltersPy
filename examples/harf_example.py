from sklearn import datasets
from NoiseFiltersPy.HARF import HARF

dataset = datasets.load_iris()
data = dataset.data
classes = dataset.target
harf = HARF()
filter = harf(data, classes)
