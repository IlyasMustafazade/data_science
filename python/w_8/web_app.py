import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


sys.path.append("/Users/YTU/source/repos/ai/ai_helper")
from matrix_helper import extract_X_y_df
from ui_helper import (load_dataset, make_slider,
                       extract_param_val_dict, make_cl,
                       display_metric_data) 
from metrics_helper import make_name_metric_dict


def main():
    st.title("Streamlit example")
    st.write(
        """
        # Explore different classifier
        """
    )

    set_file_dict = {"Water Portability": "water_portability.csv",
                                   "Diabetes": "diabetes.csv"}
    set_tpl = tuple(set_file_dict.keys())
    file_tpl = tuple(set_file_dict.values())
    name_cl_dict = {"SVM": SVC(), "KNN": KNeighborsClassifier(), "Random Forest": RandomForestClassifier()}
    cl_tpl = tuple(name_cl_dict.keys())

    chosen_set = st.sidebar.selectbox("Select Dataset", set_tpl)
    chosen_cl = st.sidebar.selectbox("Select Classifier", cl_tpl)
    df = load_dataset(chosen_set=chosen_set, set_file_dict=set_file_dict)
    df_column_arr = df.columns
    simple_imputer = SimpleImputer(strategy="mean")
    simple_imputer.fit(df)
    matrix = simple_imputer.transform(df)
    df = pd.DataFrame(data=matrix, columns=df_column_arr)
    X, y = extract_X_y_df(df=df)

    df_shape = df.shape
    y_uniq = np.unique(y)
    len_y_uniq = len(y_uniq)

    st.write("Shape of dataset: ", df_shape)
    st.write("Number of classes in target feature: ", len_y_uniq)

    cl_param_range_lst = [("KNN", "n_neighbors", (1, 15)), ("SVM", "C", (0.01, 10.0)),
                          ("Random Forest", "max_depth", (2, 15)),
                          ("Random Forest", "n_estimators", (1, 100))]

    param_val_lst = make_slider(chosen_cl=chosen_cl, cl_param_range_lst=cl_param_range_lst)
    param_val_dict = extract_param_val_dict(param_val_lst=param_val_lst)

    cl = make_cl(chosen_cl=chosen_cl, param_val_dict=param_val_dict, name_cl_dict=name_cl_dict) 
    TEST_SIZE = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    cl.fit(X_train, y_train)
    y_hat = cl.predict(X_test)
    name_metric_dict = make_name_metric_dict(y=y_test, y_hat=y_hat)
    display_metric_data(name_metric_dict=name_metric_dict)

    pca = PCA(2)
    pca.fit(X)
    X_proj = pca.transform(X)
    x1 = X_proj[:, 0]
    x2 = X_proj[:, 1]
    fig = plt.figure("Scatter Plot")
    plt.scatter(x1, x2, c=y, alpha=0.5)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(fig)


if __name__ == "__main__":
    main()














