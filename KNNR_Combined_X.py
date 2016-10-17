# CPSC483_KNNR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

#loading in training data
X = np.load("predictors.npy")
training_y_color = np.load("labels_color.npy")
training_y_quality = np.load("labels_quality.npy")
training_y_quality_binary = np.load("labels_quality_binary.npy")

#Check if data loaded in
print(X.shape)

#Test Wine
neigh = KNeighborsRegressor(n_neighbors=20)
neigh.fit(X, training_y_color)
neigh.fit(X, training_y_quality)
neigh.fit(X, training_y_quality_binary)
#test_point = np.array(X)
#test1 = test_point.reshape(1,-1)
#print(neigh.predict(test1))
