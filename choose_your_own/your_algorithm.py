#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=8)
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



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
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




from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(
    n_estimators=600,
    learning_rate=1)
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
### visualization code (prettyPicture) to show you the decision boundary








try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
