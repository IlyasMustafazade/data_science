import numpy as np
import pandas as pd
import skopt as skopt
from sklearn import (datasets, ensemble, gaussian_process, linear_model,
                     metrics, model_selection, neighbors, svm)
from skopt import space

from funcs import *


def main():
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        test_size=0.25,
                                                                        random_state=0)
    sv_cl_param_dict = {
        'C': space.Real(1e-6, 1e+6, prior='log-uniform'),
        'gamma': space.Real(1e-6, 1e+1, prior='log-uniform'),
        'degree': space.Integer(1, 8),
        'kernel': space.Categorical(['linear', 'poly', 'rbf'])
    }
    optimizer = skopt.BayesSearchCV(
        svm.SVC(), sv_cl_param_dict, n_iter=32, random_state=0)
    optimizer.fit(X_train, y_train)
    print(optimizer.score(X_test, y_test
                          ))
    print("\nBest params -> \n",
          optimizer.best_params_)


if __name__ == "__main__":
    main()
