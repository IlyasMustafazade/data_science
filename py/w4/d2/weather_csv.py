import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection


def main():

    f_space = pd.read_csv("weather.csv")

    # print(f_space.head())

    # print(f_space.shape)

    # print(f_space.isnull().sum())

    # print(f_space.drop(labels=f_space.iloc[:, -12:].columns, axis=1).describe())

    # f_space.plot(x="MinTemp", y="MaxTemp", style="x")

    # plt.xlabel("MinTemp")

    # plt.ylabel("MaxTemp")

    # plt.title("MinTemp vs MaxTemp")

    x = np.reshape(np.array(f_space["MinTemp"]), (-1, 1))

    # print(x)

    y = np.array(f_space["MaxTemp"])

    # print(y)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.15)

    regressor = sklearn.linear_model.LinearRegression()

    regressor.fit(x_train, y_train)

    f = regressor.predict(x_test)

    r_2 = sklearn.metrics.r2_score(y_test, f)

    # print("r_2 -> ", r_2)

    mse = sklearn.metrics.mean_squared_error(y_test, f)

    # print("mse -> ", mse)

    mae = sklearn.metrics.mean_absolute_error(y_test, f)

    # print("mae -> ", mae)

    rmse = np.sqrt(mse)

    # print("rmse -> ", rmse)

    pred_act_df = pd.DataFrame({"Actual": y_test, "Predicted": f})

    # plt.scatter(x_test, y_test, color="blue")

    # plt.plot(x_test, f, color="red")

    pred_act_df_head = pred_act_df.head(20)

    pred_act_df.plot(kind="bar")

    plt.show()




if __name__ == "__main__": main()



