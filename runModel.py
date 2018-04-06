import sys
from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, datasets, utils, cross_validation, \
    grid_search, svm, ensemble,neural_network,decomposition, linear_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit, KFold
# import preprocessing as W


def runTuneTest(learner, parameters, X_train,y_train,X_test, y_test):
    clf = grid_search.GridSearchCV(learner,parameters, n_jobs = -1)

    clf.fit(X_train,y_train)

    print "Best parameters: ", clf.best_params_

    accuracy = clf.score(X_test,y_test)
    trainingAccuracy = clf.score(X_train,y_train)


    return accuracy,trainingAccuracy,clf


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.001, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



def plot_learning_curve_synergy(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve_synergy(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def learning_curve_synergy(estimator, X, y, cv=None, n_jobs=None, train_sizes=None, option=0):
    train_sizes_actual = X.shape[0]*train_sizes
    train_scores_agg = []
    test_scores_agg = []

    for number in train_sizes_actual:
        X,y = utils.shuffle(X,y)
        X_sub = X[:int(number)]
        y_sub = y[:int(number)]
        train_scores = []
        test_scores = []
        for train_index, test_index in cv.split(X_sub):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # need to calculate synergy and counter after splitting the data
            X_train, chart1, chart2 = W.addSynergyCounterTrain(X_train,y_train)
            X_test = W.addSynergyCounterTest(X_test,chart1,chart2)

            if option == 4:
                # only use synergy column
                X_train = X_train[:, [-2]]
                X_test = X_test[:, [-2]]
            if option == 5:
                # only use counter column
                X_train = X_train[:, [-1]]
                X_test = X_test[:, [-1]]
            if option == 6:
                # only use synergy and counter column
                X_train = X_train[:, [-2, -1]]
                X_test = X_test[:, [-2, -1]]
            estimator.fit(X_train,y_train)
            train_score = estimator.score(X_train,y_train)
            test_score = estimator.score(X_test,y_test)
            train_scores.append(train_score)
            test_scores.append(test_score)
        train_scores_agg.append(train_scores)
        test_scores_agg.append(test_scores)

    train_scores_agg = np.array(train_scores_agg)
    test_scores_agg = np.array(test_scores_agg)
    train_scores_mean = np.mean(train_scores_agg, axis=1)
    train_scores_std = np.std(train_scores_agg, axis=1)
    test_scores_mean = np.mean(test_scores_agg, axis=1)
    test_scores_std = np.std(test_scores_agg, axis=1)

    train_scores_agg = np.array(train_scores_agg)
    test_scores_agg = np.array(test_scores_agg)
    print "train_sizes_actual: ", train_sizes_actual
    print "train_scores_mean: ", train_scores_mean
    print "train_scores_std: ", train_scores_std
    print "test_scores_mean: ", test_scores_mean
    print "test_scores_std: ", test_scores_std
    return np.array(train_sizes_actual), np.array(train_scores_agg), np.array(test_scores_agg)

def takeInput():
    while True:
        instruction = """
        Please choose a dataset, enter
        1 for original data,
        2 for PolynomialFeatures,
        3 for original data + synergy + counter,
        4 for just synergy,
        5 for just counter,
        6 for synergy and counter.
        Enter 0 to quit.
        """
        dataset = raw_input(instruction)
        dataset = int(dataset)
        if dataset == 0:
            exit()

        y_train = np.load("trainLabel.npy")
        y_test = np.load("testLabel.npy")

        if dataset == 1:
            # original data
            X_train = sparse.load_npz("trainData.npz").todense()
            X_test = sparse.load_npz("testData.npz").todense()
        elif dataset == 2:
            # PolynomialFeatures
            X_train = sparse.load_npz("trainDataadded.npz").todense()
            X_test = sparse.load_npz("testDataadded.npz").todense()
        elif dataset == 3:
            # both synergy and column added
            X_train = sparse.load_npz("trainWithBoth.npz").todense()
            X_test = sparse.load_npz("testWithBoth.npz").todense()
        elif dataset == 4:
            # synergy column added
            X_train = sparse.load_npz("trainWithCombo.npz").todense()
            X_test = sparse.load_npz("testWithCombo.npz").todense()
            # only use synergy
            X_train = X_train[:, -1]
            X_test = X_test[:, -1]
        elif dataset == 5:
            # counter column added
            X_train = sparse.load_npz("trainWithCounter.npz").todense()
            X_test = sparse.load_npz("testWithCounter.npz").todense()
            # only use counter
            X_train = X_train[:, -1]
            X_test = X_test[:, -1]
        elif dataset == 6:
            # both synergy and column added
            X_train = sparse.load_npz("trainWithBoth.npz").todense()
            X_test = sparse.load_npz("testWithBoth.npz").todense()
            # only use synergy and counter
            X_train = X_train[:, [-2, -1]]
            X_test = X_test[:, [-2, -1]]
        else:
            print "wrong input"
            exit()
        X_train = X_train.astype(np.float)
        X_test = X_test.astype(np.float)
        y_train = y_train.astype(np.float)
        y_test = y_test.astype(np.float)

        model = raw_input("""Which model would you like to run? Please enter
        1 for Logistic Regression
        2 for SVM
        """)
        model = int(model)

        if model == 1:
            clf = linear_model.LogisticRegression()
            param = [
               {'C': [0.000001,0.001,0.1,1, 10], 'max_iter' : [50,100,1000,],'solver':['sag', 'newton-cg','lbfgs']},
               {'C': [0.000001,0.001,0.1,1, 10], 'solver':['liblinear']},
              ]
        elif model == 2:
            clf = svm.SVC()
            param = [
              {'C': [1, 10, 100], 'kernel': ['linear']},
              {'C': [1, 10, 100], 'gamma': [0.001, 0.0001,1], 'kernel': ['rbf']},
              {'C': [1, 10, 100], 'degree': [2, 3], 'kernel': ['poly']}
             ]
        accuracy,trainScore, clf = runTuneTest(clf,param,X_train,y_train,X_test,y_test)
        print "Training acc: ", trainScore
        print "Accuracy: ", accuracy

        curve = raw_input("""Would you like to run the learning curve algorithm? Please enter
        1 for yes
        2 for no
        0 to quit
        """)
        curve = int(curve)

        if curve == 1:
            fileName = raw_input("Please enter the name for the learning curve file: ")
            title = fileName
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
            estimator = clf
            if dataset == 1 or dataset == 2:
                plot = plot_learning_curve(estimator, title, X_train, y_train, ylim=(0, 1), cv=cv, n_jobs=-1)
            else:
                X_train = sparse.load_npz("trainData.npz").todense()
                X_test = sparse.load_npz("testData.npz").todense()
                plot = plot_learning_curve_synergy(estimator, title, X_train, y_train, cv=cv, option=dataset)
            plot.savefig(fileName+".png")
        elif curve == 0:
            exit()


takeInput()
