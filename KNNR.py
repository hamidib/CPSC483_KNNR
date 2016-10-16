# CPSC483_KNNR

from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import urllib
import urllib.request

# download the redwine file
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
raw_data = urllib.request.urlopen(url)
raw_data.readline()

# load the redwine CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=";")
print(dataset.shape)
# separate the redwine data from the target attributes
trainingx = dataset[:,0:11]
trainingy = dataset[:,11]

# download the whitewine file
url2 = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
raw_data2 = urllib.request.urlopen(url2)
raw_data2.readline()

# load the whitewine CSV file as a numpy matrix
dataset2 = np.loadtxt(raw_data2, delimiter=";")
print(dataset2.shape)
# separate the whitewine data from the target attributes
x2 = dataset2[:,0:11]
y2 = dataset2[:,11]

X = dataset
neigh = KNeighborsRegressor(n_neighbors=20)
neigh.fit(trainingx,trainingy)

#test wine
testPoint = np.array(dataset[])
test1 = testPoint.reshape(1,-1)
print(neigh.predict(test1))

#url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

#raw_data = urllib.urlopen(url)

#dataset = np.loadtxt(raw_data, delimiter=",")
#print(dataset.shape)

#X = dataset[:,0:7]
#y = dataset[:,8]
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
#distances, indices = nbrs.kneighbors(X)
