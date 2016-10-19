# CPSC483_KNNR
import pandas as pd
import numpy as np

#X = np.load("predictors.npy")

#training_y_color = np.load("labels_color.npy")

#training_y_quality = np.load("labels_quality.npy")

#training_y_quality_binary = np.load("labels_quality_binary.npy")

#test_predictors_y  = np.load("test_predictors.npy")

#training_predictors_y = np.load("labels_color.npy")

#px2 = np.reshape((-1,6499))
#px2 = np.reshape((-1, 71489)
#df = pd.DataFrame({'X':px2[:,0],'Y':px2[:,1],'Z':px2[:,2],'A':px2[:,3],'B':px2[:,4],'C':px2[:,5],'D':px2[:,6],'E':px2[:,7],'F':px2[:,8],'G':px2[:,9],'H':px2[:,10]})
#pd.DataFrame(data=data[1:,1:],index=data[1:,0],columns=data[0,1:]) 
# Present the dataset
print("Red Wine Data")
df1 = pd.read_csv('winequality-red.csv', sep = ';')
df1.describe()
df1.corr()
# Present the dataset
print("White Whine Data")
df2 = pd.read_csv('winequality-red.csv', sep = ';')
df2.describe()
df2.corr()
#Concatenating white and red wine data sets http://pandas.pydata.org/pandas-docs/stable/merging.html
frames = [df1, df2]
df = pd.concat(frames)

#Split the data into training and testing sets
from sklearn.cross_validation import train_test_split
Features = df[list(df.columns)[:-1]]
Quality = df['quality']
Features_train, Features_test, Quality_train, Quality_test = train_test_split(Features, Quality)

#Create and fit the model on the training data
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=20)
neigh.fit(Features_train, Quality_train)

#neigh.fit(X, training_y_quality)
#neigh.fit(X, training_y_quality_binary)

#Evaluate the predictions of the model
Quality_predictions = neigh.predict(Features_test)

# Create scatterplot of Predicted Quality against True Quality 
import matplotlib.pylab as plt
plt.scatter(Quality_test, Quality_predictions)
plt.xlabel('True Quality')
plt.ylabel('Predicted Quality')
plt.title('Predicted Quality Against True Quality ')
plt.show()
