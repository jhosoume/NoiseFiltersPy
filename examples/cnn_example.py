from sklearn import datasets

import context
from NoiseFiltersPy.CNN import CNN

dataset = datasets.load_iris()
data = dataset.data
classes = dataset.target
cnn = CNN()
filter = cnn(data, classes)
