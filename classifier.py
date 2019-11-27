import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class Classifier:
    def __init__(self, clf):
        self.clf = clf

    def train(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, X, y, metric):
        """
        Possible evaluation metrics:
        1. accuracy
        2. precision
        3. recall
        4. auc
        5. default: error = 1 - accuracy
        """
        p_pred = self.clf.predict_proba(X)[:, 1]  # probability
        # b_pred = np.where(p_pred > 0.5, 1, 0)     # binary 
        b_pred = self.clf.predict(X)              # binary 
        if metric == 'accuracy':
            score = accuracy_score(y, b_pred)
        elif metric == 'precision':
            score = precision_score(y, b_pred)
        elif metric == 'recall':
            score = recall_score(y, b_pred)
        elif metric == 'auc':
            score = roc_auc_score(y, p_pred)
        else:  # default is error rate
            score = 1 - accuracy_score(y, b_pred)
        return score
