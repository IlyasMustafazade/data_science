import numpy as np
import pandas as pd
from sklearn import (ensemble, gaussian_process, linear_model, metrics,
                     model_selection, neighbors, svm)

from funcs import *


def objective(x, noise=0):
    noise = np.random.normal(loc=0, scale=noise)
    return (x**2 * math.sin(5 * math.pi * x)**6.0) + noise


def main():
    X = np.random.random(100)
    y = np.array([objective(x) for x in X])
    X = np.reshape(X, (-1, 1))
    y = np.reshape(y, (-1, 1))
    model = gaussian_process.GaussianProcessRegressor()
    model.fit(X, y)
    plot_real_surrogate(X, y, model)
    bayesian_optimize(X, y, objective, model)
    plt.show()


if __name__ == "__main__":
    main()
