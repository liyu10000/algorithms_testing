import os
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from classifier import Classifier
from testing import McNemarTest, CV52PairedTTest



data = pd.read_csv('matrix.txt', sep=',', header=None, engine='python')
X = data.values[:,2:]
y = data.values[:,1]
print(X.shape, y.shape)

clf1 = Classifier(XGBClassifier(n_jobs=8))
clf2 = Classifier(RandomForestClassifier(n_estimators=20, n_jobs=8))
clf3 = Classifier(SVC(kernel='linear', C=1.0, probability=True))
clf4 = Classifier(LogisticRegression(penalty='l2', n_jobs=8))

clfs = [clf1, clf2, clf3, clf4]
names = ['XGB', 'RF', 'SVM', 'LR']
results = {}

for i in range(len(clfs)):
    results[names[i]] = {}
    for j in range(i+1, len(clfs)):
        clf1 = clfs[i]
        clf2 = clfs[j]

        print("Testing {} vs. {}".format(names[i], names[j]))
        print("\nMcNemar's test:")
        rejectMN, score1MN, score2MN = McNemarTest(clf1, clf2, X, y)
        if rejectMN:
            print('reject null hypothesis')
        else:
            print('fail to reject null hypothesis: clf1 and clf2 make errors in the same way')


        print("\n5x2 cv paired t test:")
        rejectCV, score1CV, score2CV = CV52PairedTTest(clf1, clf2, X, y)
        if rejectCV:
            print('reject null hypothesis')
        else:
            print('fail to reject null hypothesis: clf1 and clf2 make errors in the same way')

        results[names[i]][names[j]] = {'McNemar':[rejectMN, score1MN, score2MN], '52CV':[rejectCV, score1CV, score2CV]}


print()
pprint(results)


# save results
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)