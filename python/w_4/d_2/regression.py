import matplotlib.pyplot as plt
import numpy as np
import seaborn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection


def main():

    sample_size = 25

    # create matrix of 1 column vector representing x-axis values

    x = np.reshape(np.arange(sample_size), (sample_size, 1))

    # create row vector representing y-axis values

    y = np.random.randint(0, high=500, size=(sample_size,))

    # split x, y axes to train and test

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    regressor = sklearn.linear_model.LinearRegression()

    regressor.fit(x_train, y_train)

    estimated_func = regressor.coef_ * x_test + regressor.intercept_

    y_pred = regressor.predict(x_test)

    plt.plot(x_test, y_pred, "purple")

    plt.plot(x_test, y_test, "green")

    # Model estimations

    r_2 = sklearn.metrics.r2_score(y_test, y_pred)

    # print("r_2 -> ", r_2)

    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)

    # print("mse -> ", mse)

    mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)

    # print("mae -> ", mae)

    rmse = np.sqrt(mse)

    # print("rmse -> ", rmse)

    plt.show()


if __name__ == "__main__": main()


