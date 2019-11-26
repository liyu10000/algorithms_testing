import math
import numpy as np
import statsmodels
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold


# def McNemar(A, B):
#     """ hand written McNemar's test
#     :param A: binary prediction results from classifier A, 1D array
#     :param B: binary prediction results from classifier B, 1D array
#     :return: statistic
#     """
#     diff = A - B
#     n01 = np.count_nonzero(diff == 1)
#     n10 = np.count_nonzero(diff == -1)
#     print(n01, n10)
#     return (abs(n01 - n10) - 1) ** 2 / (n01 + n10)


def McNemar(A, B):
    """ use McNemar's test from statsmodels
    :param A: binary prediction results from classifier A, 1D array
    :param B: binary prediction results from classifier B, 1D array
    :return: statistic, pvalue
    """
    AB = np.vstack((A, B)).T
    table = sm.stats.Table.from_data(AB)
    bunch = statsmodels.stats.contingency_tables.mcnemar(table.table_orig.values, exact=False, correction=True)
    rslt = table.test_nominal_association()
    return bunch.statistic, rslt.pvalue


def McNemarTest(clf1, clf2, X, y, alpha=0.05):
    """ use McNemar's test from statsmodels
    :param clf1: 1st classifier, Classifier object
    :param clf2: 2nd classifier, Classifier object
    :param X: features X
    :param y: labels y
    :param alpha: significance level
    :return: decision, reject null hypothesis or not
    """
    test_size = 0.2
    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    clf1.train(X_train, y_train)
    clf2.train(X_train, y_train)
    A = clf1.predict(X_test)
    B = clf2.predict(X_test)

    # compute statistic
    statistic, pvalue = McNemar(A, B)

    # get chi square, given alpha and df = 1
    df = 1
    chi_square = stats.chi2.ppf(1-alpha, df)
    print('calculated chi square statistic: {:.4f}, chi square value under alpha={}: {:.4f}, p-value: {:.4f}'.format(statistic, alpha, chi_square, pvalue))

    # decision: reject H0 if reject = True
    reject = statistic > chi_square
    return reject


def CV52PairedTTest(clf1, clf2, X, y, alpha=0.05):
    """ use McNemar's test from statsmodels
    :param clf1: 1st classifier, Classifier object
    :param clf2: 2nd classifier, Classifier object
    :param X: features X
    :param y: labels y
    :param alpha: significance level
    :return: decision, reject null hypothesis or not
    """
    # set 5 seeds for each 2-fold cross validation
    seeds = [5, 13, 20, 42, 68]
    # place holder for the score difference of the 1st fold of 1st cv
    p11 = 0.0
    # initialize the variance estimate
    s_sqr = 0.0

    for i_s, seed in enumerate(seeds):
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        pi = np.zeros(2)
        for i_f, (train_idx, test_idx) in enumerate(folds.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            clf1.train(X_train, y_train)
            clf2.train(X_train, y_train)
            score1 = clf1.score(X_test, y_test, metric='error')
            score2 = clf2.score(X_test, y_test, metric='error')
            pi[i_f] = score1 - score2
            if i_s == 0 and i_f == 0:
                p11 = pi[0]
        p_bar = (pi[0] + pi[1]) / 2
        s_sqr += (pi[0] - p_bar) ** 2 + (pi[1] - p_bar) ** 2

    # compute t statistic
    t_tilde = p11 / (s_sqr / 5) ** 0.5
    # get t value, under significance lavel alpha=0.05, two-tail test
    df = 5 # degree of freedom
    t_value = stats.t.ppf(1-alpha/2, df)
    print('calculated t statistic: {:.4f}, t value under alpha={}: {:.4f}'.format(t_tilde, alpha, t_value))

    # decision: reject H0 if reject = True
    reject = t_tilde >= t_value or t_tilde <= -t_value
    return reject

