import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, model_selection
import matplotlib.pyplot as plt

def main():

    stock_market = {'Year': [2017,2017,2017,2017,2017,2017,
    2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,
                1,12,11,10,9,8,7,6,5,4,3,2,1],
                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,
                2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,
                5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'Stock_Index_Price': [1464,1394,1357,1293,
                1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]        
                }

    f_space = pd.DataFrame(data=stock_market)

    X = f_space[["Unemployment_Rate", "Interest_Rate"]]

    Y = f_space[["Stock_Index_Price"]]

    # print("X -> ", X)

    # print("Y -> ", Y)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

    regressor = linear_model.LinearRegression()

    regressor.fit(X_train, Y_train)
    
    f = regressor.predict(X_test)

    Y_test = np.array(Y_test).flatten()

    f = f.flatten()

    pred_act_df = pd.DataFrame(data={"Actual": Y_test, "Predicted": f})

    pred_act_df.plot(kind="bar")

    plt.show()


if __name__ == "__main__": main()


