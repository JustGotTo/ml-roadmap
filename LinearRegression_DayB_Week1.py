### In this project I will try to implement the LinearRegression using only NumPy
# Dataset used - California Housing from sklearn

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X = np.array(fetch_california_housing()['data'])
Y = np.array(fetch_california_housing()['target'])
#setting up the data to work with

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #splitting data to perform a test with loss function
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test] #adding a column of ones as a bias

theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train) #A vector of features for linear regression

meants = np.mean(Y_test) #getting mean dependent variables of test data

def rsquared(tsdata, args, actual, mean):
    prediction = np.dot(tsdata, args) #Predicted variables using vector of features
    rss = np.linalg.norm(prediction - actual)**2 #Residual sum of squares
    tss = np.linalg.norm(actual - mean)**2 #Total sum of squares
    return 1 - rss/tss
r2 = rsquared(X_test, theta, Y_test, meants)
#function for giving the accuracy of the predicted features of linear regression, could be given as list of instructions instead because it is only called once
print(r2)
#printing out the loss function to determine the accuracy


