#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clf = tree.DecisionTreeClassifier(min_samples_split=40)
print(len(features_train[0]))
print("Fitting of data has been started")
time1 = time()
clf.fit(features_train, labels_train)
print("time of fitting: ",round(time()-time1,3))


print("Prediction of data has been started")
time2 = time()
clf.predict(features_test)
print("time of predicting: ",round(time()-time2,3))
print()
print()

print("Scoring of data has been started")
time3 = time()
print("Accuracy = ",clf.score(features_test,labels_test))
print("time of scoring: ",round(time()-time3,3))
#########################################################


