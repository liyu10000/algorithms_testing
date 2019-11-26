import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from classifier import Classifier
from testing import McNemarTest, CV52PairedTTest
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('matrix.txt',sep=',',header=None,engine='python')
X = data.values[:,2:]
y = data.values[:,1]
print(X.shape,y.shape)

clf1 = Classifier(XGBClassifier(n_jobs=8))
clf2 = Classifier(RandomForestClassifier(n_estimators=20, n_jobs=8))
clf3 = Classifier(SVC(kernel='linear',C=1.0))
clf4 = Classifier(SVC(kernel='rbf',C=1.0))
clf5 = Classifier(LogisticRegression(penalty='l2'))

clfs = [clf1,clf2,clf3,clf4,clf5]

for i in range(len(clfs)):
    for j in range(i+1,len(clfs)):
        clf1 = clfs[i]
        clf2 = clfs[j]

        print("McNemar's test:")
        reject = McNemarTest(clf1, clf2, X, y)
        if reject:
            print('reject null hypothesis')
        else:
            print('fail to reject null hypothesis: clf1 and clf2 make errors in the same way')


        print("5x2 cv paired t test")
        reject = CV52PairedTTest(clf1, clf2, X, y)
        if reject:
            print('reject null hypothesis')
        else:
            print('fail to reject null hypothesis: clf1 and clf2 make errors in the same way')