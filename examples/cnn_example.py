from sklearn import datasets
from NoiseFiltersPy.CNN import CNN

dataset = datasets.load_iris()
data = dataset.data
classes = dataset.target
cnn = CNN()
filter = cnn(data, classes)
