import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, neighbors, svm


def main():
    file_base = "hpt_small"
    file_ext = ".csv"
    file_name = file_base + file_ext
    f_space = pd.read_csv(file_name)
    f_space = pd.get_dummies(f_space, drop_first=True)
    f_space_columns = f_space.columns
    predictor_vector = f_space.iloc[:-1]
    dependent_feature = f_space.iloc[:-1]
    predictor_vector_train, predictor_vector_test, \
        dependent_feature_train, dependent_feature_test = \
        model_selection.train_test_split(
            predictor_vector, dependent_feature, test_size=0.2)
    n_neighbors = np.arange(1, 31)
    cl_arr = []
    for n in n_neighbors:
        classifier = neighbors.KNeighborsClassifier(n_neighbors=n)
        cl_arr.append(classifier)
    cl_arr = np.array(cl_arr)
    predictor_arr = f_space_columns[:-1]
    predictor_arr = np.array(predictor_arr)
    outcome_feature = f_space_columns[-1]
    outcome_feature = np.array(outcome_feature)
    for model in cl_arr:
        train_eval_model(
            model, f_space, predictor_arr, outcome_feature)

    # hey hey


if __name__ == "__main__":
    main()
