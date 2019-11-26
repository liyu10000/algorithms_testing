import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from classifier import Classifier
from testing import McNemarTest, CV52PairedTTest


X = np.random.random((1000, 50))
y = np.random.randint(2, size=1000)
print(X.shape, y.shape)

clf1 = Classifier(XGBClassifier(n_jobs=8))
clf2 = Classifier(RandomForestClassifier(n_estimators=20, n_jobs=8))

decision = McNemarTest(clf1, clf2, X, y)
if decision:
    print('reject null hypothesis')
else:
    print('fail to reject null hypothesis: clf1 and clf2 make errors in the same way')

decision = CV52PairedTTest(clf1, clf2, X, y)
if decision:
    print('reject null hypothesis')
else:
    print('fail to reject null hypothesis: clf1 and clf2 make errors in the same way')