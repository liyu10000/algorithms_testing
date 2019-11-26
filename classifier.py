from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class Classifier:
    def __init__(self):
        self.clf = None

    def train(self, X, y):
        self.clf(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, X, y, metric='error'):
        """
        Possible evaluation metrics:
        1. accuracy
        2. error = 1 - accuracy
        3. precision
        4. recall
        5. auc
        """
        b_pred = self.clf.predict(X)        # binary 
        p_pred = self.clf.predict_proba(X)  # probability
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
