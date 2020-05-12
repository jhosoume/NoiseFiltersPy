from scipy.io import arff as arff_io
import pandas as pd
from NoiseFiltersPy.HARF import HARF

dataset = arff_io.loadarff("18_mfeat-morphological.arff")
import pdb; pdb.set_trace()
data = pd.DataFrame(dataset[0])
data = dataset.data
classes = dataset["class"].values
harf = HARF()
filter = harf(data, classes)
