import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics, model_selection, linear_model

def main(): 

    f_space = pd.read_csv("pima-indians-diabetes.csv")
    
    with open("standardize.xlsx", "wb") as f: 

        standardize(f_space).to_excel(f)



def standardize(f_space):

    return pd.DataFrame(data=preprocessing.StandardScaler().fit_transform(f_space))


if __name__ == "__main__": main()


