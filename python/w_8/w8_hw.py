import re

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn import over_sampling
from sklearn import (compose, ensemble, impute, metrics, model_selection,
                     pipeline, preprocessing)


def main():
    file_obj = "income_evaluation"
    file_ext = ".csv"
    full_name = file_obj + file_ext
    df = pd.read_csv(full_name)
    df = df.rename(str.strip, axis="columns")
    bad_columns = np.array([
        "capital-gain", "capital-loss", "education", "fnlwgt", "native-country"])
    df = df.drop(labels=bad_columns, axis=1)
    df = df.replace(to_replace=re.compile(
        r" \?"), value=np.nan, regex=True)
    df = df.dropna()

    cont_cols = ["age", "hours-per-week"]
    binary_cols = ["workclass",
                   "marital-status", "occupation", "relationship", "race"]
    ordinal_cols = [
        "education-num", "sex", "income"]
    df_columns = df.columns

    log_transformer = preprocessing.FunctionTransformer(
        func=np.log10)
    imputer_pipe = pipeline.make_pipeline(impute.SimpleImputer(
        missing_values=r" \?", strategy="most_frequent"))
    cont_pipe = pipeline.make_pipeline(
        log_transformer, preprocessing.Normalizer())
    binary_pipe = pipeline.make_pipeline(
        ce.BinaryEncoder())
    ordinal_pipe = pipeline.make_pipeline(
        ce.OrdinalEncoder())

    col_transformer = compose.make_column_transformer(
        (cont_pipe,
         cont_cols), (binary_pipe, binary_cols),
        (ordinal_pipe, ordinal_cols)
    )
    col_transformer.fit(df)
    df = col_transformer.transform(df)
    matrix = df

    sampler = over_sampling.SMOTE()

    m, n = np.shape(matrix)
    X = np.zeros(shape=(m, n-1))
    y = np.zeros(shape=(m,))
    for index, row_vector in enumerate(matrix):
        X[index] = row_vector[:-1]
        y[index] = row_vector[-1] - 1

    X, y = sampler.fit_resample(X, y)

    test_size = 0.2
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size)
    random_forest_cl = ensemble.RandomForestClassifier()
    random_forest_cl.fit(X_train, y_train)
    y_hat = random_forest_cl.predict(X_test)
    with open("w8_hw.txt", "w") as f:
        comp_metrics(
            y=y_test, y_hat=y_hat, file_obj=f)


def comp_metrics(y=None, y_hat=None, file_obj=None, show_metrics=True):
    accuracy = metrics.accuracy_score(
        y, y_hat)
    f1_score = metrics.f1_score(
        y, y_hat, zero_division=1)
    f2_score = metrics.fbeta_score(
        y, y_hat, beta=2, zero_division=1)
    conf_matrix = metrics.confusion_matrix(
        y, y_hat)
    cl_report = metrics.classification_report(
        y, y_hat, zero_division=1)
    roc_auc_score = metrics.roc_auc_score(
        y, y_hat)
    name_metric_dict = {"Accuracy score": accuracy, "F1 score": f1_score,
                        "F2 score": f2_score, "Confusion matrix": conf_matrix,
                        "Classification report": cl_report, "ROC-AUC score": roc_auc_score}
    if show_metrics is True:
        for name, metric in name_metric_dict.items():
            print("\n", name, "->\n\n",
                  metric, file=file_obj)
    as_tuple = tuple(name_metric_dict.values())
    return as_tuple


if __name__ == "__main__":
    main()
