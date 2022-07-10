import numpy as np
import pandas as pd
from sklearn.preprocessing import Binarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer


def main():

    f_space = pd.read_csv("pima-indians-diabetes.csv")

    with open("binarize.xlsx", "wb") as f: 

        one_hot_encode(f_space, 0).to_excel(f)


def one_hot_encode(f_space, column_index):

    return pd.DataFrame(data=ColumnTransformer([("one_hot_encoder",
                OneHotEncoder(categories="auto"), [column_index])],
                    remainder="passthrough").fit_transform(f_space))


def add_binary_pd(keys_lst, vals_lst, df):

    copy_df = df

    for i in range(len(keys_lst)):

        key, val = keys_lst[i], vals_lst[i]

        copy_df["Binary " + key + " > " + str(val)] = pd.Series(copy_df[key] > val).astype("int64")
    
    return copy_df


def add_binary_sk(keys_lst, vals_lst, df):

    copy_df = df

    reshaped_lst = np.array([list(copy_df[i]) for i in keys_lst])

    reshaped_lst, transformer_lst = np.array([np.reshape(i, (1, -1)) for i in reshaped_lst]), \
                                    np.array([Binarizer(threshold=i) for i in vals_lst])

    for i in range(len(transformer_lst)):

        key, val = keys_lst[i], vals_lst[i]

        transformer, reshaped_col = transformer_lst[i], reshaped_lst[i]

        copy_df["Binary " + key + " > " + str(val)] = pd.Series(transformer.transform(reshaped_col)[0])

    return copy_df


if __name__ == "__main__": main()


