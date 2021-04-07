from sklearn import datasets
from NoiseFiltersPy.AENN import AENN

dataset = datasets.load_iris()
data = dataset.data
classes = dataset.target
aenn = AENN()
filter = aenn(data, classes)
print(filter.clean_data)