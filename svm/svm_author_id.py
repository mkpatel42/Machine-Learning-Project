#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



from sklearn import svm
clf = svm.SVR(gamma='auto',kernel='rbf',C=100)
print()
print()
print("Fitting of data has been started")
time1 = time()
clf.fit(features_train, labels_train)
print("time of fitting: ",round(time()-time1,3))
print()
print()

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


# For kernel = 'rbf' and c = 1000
# time of fitting:  650.078
# Accuracy =  0.8686059539191174
# time of predicting:  63.199






#########################################################
### your code goes here ###

#########################################################


