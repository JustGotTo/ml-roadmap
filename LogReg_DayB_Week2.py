# Engineering task for Day B, week 2
#Create a logistic regression model from scratch, using NumPy and an Iris dataset from sklearn and testing it using binary cross-entropy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = np.array(load_iris().data)
#loading data for the model
target = np.array(load_iris().target)
#loading the outputs for data

#binary classification
rem = target != 2
data = data[rem]
target = target[rem]


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
#splitting data into test and train
X_train = np.c_[X_train, np.ones(X_train.shape[0])]
X_test = np.c_[X_test, np.ones(X_test.shape[0])]

weights = np.zeros(X_train.shape[1]) #weights parameter
lr = 0.01
#implementing sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#modifying the weights using a gradient descend
for epoch in range(0,1000):
    ans = X_train.dot(weights)
    pred = sigmoid(ans)
    gradient = np.dot(X_train.T, (pred - y_train))/len(X_train)
    weights -= gradient * lr

print(f"Weights after epoch 1000: {weights}")
predictions = sigmoid(X_test.dot(weights)) > 0.5 #outputting predictions with threshold set to 0.5
print(f"Predictions: {np.int32(predictions)}")
accuracy = (predictions == y_test).mean() #Giving the accuracy of the model. Since there very few test data, the model shows accuracy of 100%, as every prediction was true
print(f"Accuracy: {accuracy}")
