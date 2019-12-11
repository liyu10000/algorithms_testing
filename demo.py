import os
import time
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


def news():
    data = pd.read_csv('matrix.txt', sep=',', header=None, engine='python')
    X = data.values[:,2:]
    y = data.values[:,1]
    return X, y


def wine():
    winenames = ['label','Alcohol','Malic acid','Ash','Alcalinity of ash',
                'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
                'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines',
                'Proline']
    wine = pd.read_csv('wine.data',sep=',',header=None, names=winenames, engine='python')
    X = wine.values[:, 1:14]
    y = wine.values[:,0]
    return X, y


def santander():
    df = pd.read_csv('./santander/train.csv')
    xdf = df.drop(['ID_code', 'target'], axis=1)
    ydf = df['target']
    # check pandas version
    v = pd.__version__
    v = int(v.split('.')[1])
    if v < 24: # before version '0.24.0'
        X = xdf.values
        y = ydf.values
    else:
        X = xdf.to_numpy()
        y = xdf.to_numpy()
    return X, y


def classifiers():
    clf1 = Classifier(XGBClassifier(n_jobs=8))
    clf2 = Classifier(RandomForestClassifier(n_estimators=20, n_jobs=8))
    clf3 = Classifier(SVC(kernel='linear', C=1.0, probability=True))
    clf4 = Classifier(LogisticRegression(penalty='l2', n_jobs=8))

    clfs = [clf1, clf2, clf3, clf4]
    names = ['XGB', 'RF', 'SVM', 'LR']
    return clfs, names


def test(clfs, names, X, y):
    results = {}

    for i in range(len(clfs)):
        results[names[i]] = {}
        for j in range(i+1, len(clfs)):
            clf1 = clfs[i]
            clf2 = clfs[j]

            print("\n\nTesting {} vs. {}".format(names[i], names[j]))
            print("\nMcNemar's test:")
            t1MN = time.time()
            rejectMN, score1MN, score2MN = McNemarTest(clf1, clf2, X, y)
            t2MN = time.time()
            if rejectMN:
                print('reject null hypothesis')
            else:
                print('fail to reject null hypothesis: clf1 and clf2 make errors in the same way')


            print("\n5x2 cv paired t test:")
            t1CV = time.time()
            rejectCV, score1CV, score2CV = CV52PairedTTest(clf1, clf2, X, y)
            t2CV = time.time()
            if rejectCV:
                print('reject null hypothesis')
            else:
                print('fail to reject null hypothesis: clf1 and clf2 make errors in the same way')

            results[names[i]][names[j]] = {'McNemar':[rejectMN, score1MN, score2MN, t2MN-t1MN], 
                                           '52CV':[rejectCV, score1CV, score2CV, t2CV-t1CV]}

    return results


def save_results(name, results):
    with open(name, 'wb') as f:
        pickle.dump(results, f)


def load_results(name):
    with open(name, 'rb') as f:
        results = pickle.load(f)
    return results



if __name__ == '__main__':
    # X, y = news()
    # X, y = santander()
    X, y = wine()
    print(X.shape, y.shape)

    clfs, names = classifiers()

    results = test(clfs, names, X, y)
    pprint(results)

    name = './wine.pkl'
    save_results(name, results)


